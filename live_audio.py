"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import time
from collections import deque

import cv2
import pyaudio
import PIL.Image
import mss

import argparse
import json

from google import genai



FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

DEFAULT_MODE = "none"

api_key = json.load(open(".api_key.json"))["api_key"]
voice_client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})


CONFIG = {"response_modalities": ["AUDIO"]}

pya = pyaudio.PyAudio()




class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, enable_downstream=False):
        self.video_mode = video_mode
        self.enable_downstream = enable_downstream  # Flag to enable downstream processing

        self.audio_in_queue = None
        self.out_queue = None
        self.combined_audio_queue = None  # Queue for combined audio
        self.downstream_queue = None  # Queue for downstream processing

        self.session = None
        self.downstream_model = None  # Placeholder for downstream model
        
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.collected_audio_data = []  # List to store collected audio chunks
        
        # For downstream processing
        self.buffer_size = 5  # Number of chunks to buffer before processing
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.last_process_time = 0
        self.process_interval = 0.5  # Process every half second

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def process_audio_for_downstream(self):
        """Process combined audio and send to downstream model in real-time."""
        print("Starting downstream audio processing...")
        
        # Initialize the downstream model here if needed
        # This is a placeholder - replace with actual model initialization
        # self.downstream_model = genai.GenerativeModel(...)
        
        while True:
            # Get audio chunk from the combined queue
            audio_chunk = await self.combined_audio_queue.get()
            
            # Add to buffer and also store in collected data
            self.audio_buffer.append(audio_chunk)
            self.collected_audio_data.append(audio_chunk)
            
            # Also put in downstream queue for potential consumers
            await self.downstream_queue.put(audio_chunk)
            
            # Process periodically to avoid overwhelming the model
            current_time = time.time()
            if current_time - self.last_process_time >= self.process_interval and len(self.audio_buffer) >= 2:
                self.last_process_time = current_time
                
                # Process the buffered audio
                await self.send_to_downstream_model()
                
    async def send_to_downstream_model(self):
        """Send buffered audio to a downstream model for sentiment analysis."""
        # This is where you would implement the connection to another Gemini model
        # For now, we'll just print the buffer info
        
        user_chunks = [chunk for chunk in self.audio_buffer if chunk['source'] == 'user']
        model_chunks = [chunk for chunk in self.audio_buffer if chunk['source'] == 'model']
        
        print(f"\nProcessing audio batch: {len(user_chunks)} user chunks, {len(model_chunks)} model chunks")
        
        # Example of how you might prepare data for a downstream model
        # Replace this with actual implementation
        """
        # Example pseudocode for calling a downstream model:
        if self.downstream_model:
            try:
                # Prepare audio data for the model (might need to convert to appropriate format)
                # This might involve combining chunks, resampling, etc.
                response = await self.downstream_model.generate_content(
                    contents=[{
                        "parts": [
                            {"text": "Analyze the sentiment and emotion in this conversation:"},
                            {"audio": {'mime_type': 'audio/pcm', 'data': processed_audio}}
                        ]
                    }]
                )
                sentiment_result = response.text
                print(f"Sentiment analysis: {sentiment_result}")
            except Exception as e:
                print(f"Error in downstream processing: {e}")
        """
        
        # Clear buffer after processing
        self.audio_buffer.clear()

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            # Create user audio chunk
            user_audio_chunk = {
                "source": "user", 
                "data": data, 
                "rate": SEND_SAMPLE_RATE,
                "timestamp": time.time()
            }
            
            # Put user audio into combined queue
            await self.combined_audio_queue.put(user_audio_chunk)
            
            # Also put into output queue for sending to model
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            
            # Create model audio chunk
            model_audio_chunk = {
                "source": "model", 
                "data": bytestream, 
                "rate": RECEIVE_SAMPLE_RATE,
                "timestamp": time.time()
            }
            
            # Put model audio into combined queue 
            await self.combined_audio_queue.put(model_audio_chunk)
            
            # Play the audio
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                voice_client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.combined_audio_queue = asyncio.Queue()  # Initialize combined queue
                self.downstream_queue = asyncio.Queue()  # Queue that external consumers can access
                
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Add the downstream processing task if enabled
                if self.enable_downstream:
                    tg.create_task(self.process_audio_for_downstream())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            print("\nUser exited. Collecting audio data...")
            pass # Proceed to finally block
        except ExceptionGroup as EG:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)
        finally:
            # Ensure audio stream is closed if it exists and is open
            if hasattr(self, 'audio_stream') and self.audio_stream and not self.audio_stream.is_stopped():
                 self.audio_stream.stop_stream()
                 self.audio_stream.close()
            # Collect all remaining audio data
            if self.combined_audio_queue:
                while not self.combined_audio_queue.empty():
                    self.collected_audio_data.append(self.combined_audio_queue.get_nowait())
            print(f"Collected {len(self.collected_audio_data)} audio chunks.")
            return self.collected_audio_data # Return the collected data


if __name__ == "__main__":
    #list_available_models() # Call the updated function


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--downstream",
        action="store_true",
        help="Enable downstream model processing",
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode, enable_downstream=args.downstream)
    collected_audio = asyncio.run(main.run())

    # Now you can process the collected_audio list
    # Each item is a dict: {"source": "user/model", "data": bytes, "rate": int, "timestamp": float}
    if collected_audio:
        print(f"Successfully collected audio data. Number of chunks: {len(collected_audio)}")
    else:
        print("No audio data was collected.")
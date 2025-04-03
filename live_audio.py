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
import multiprocessing
import queue
import threading
import json

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai


class SentimentAnalysisProcess(multiprocessing.Process):
    """A separate process for sentiment analysis to avoid interfering with the voice conversation."""
    
    def __init__(self, api_key, sentiment_model_name, audio_queue):
        super().__init__()
        self.api_key = api_key
        self.sentiment_model_name = sentiment_model_name
        self.audio_queue = audio_queue
        self.sentiment_history = multiprocessing.Manager().list()  # Shared list for sentiment results
        self.exit_flag = multiprocessing.Event()
        
    def run(self):
        """Main process function that runs sentiment analysis on audio chunks."""
        print("Starting sentiment analysis process...")
        
        # Initialize the client
        self.sentiment_client = genai.Client(api_key=self.api_key)
        
        # Buffer for collecting audio chunks
        audio_buffer = []
        last_process_time = 0
        chunk_duration = 10  # 10 seconds per chunk
        
        while not self.exit_flag.is_set():
            try:
                # Try to get audio data with a timeout to allow checking the exit flag
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    audio_buffer.append(audio_chunk)
                except queue.Empty:
                    # No data available, just continue
                    pass
                
                # Process periodically if we have enough data
                current_time = time.time()
                if current_time - last_process_time >= chunk_duration and len(audio_buffer) >= 2:
                    last_process_time = current_time
                    
                    # Process the audio in a separate thread to avoid blocking
                    analysis_thread = threading.Thread(
                        target=self.analyze_audio_sentiment,
                        args=(audio_buffer.copy(),)
                    )
                    analysis_thread.daemon = True
                    analysis_thread.start()
                    
                    # Clear the buffer after starting analysis
                    audio_buffer = []
                    
            except Exception as e:
                print(f"Error in sentiment analysis process: {e}")
                traceback.print_exc()
        
        print("Sentiment analysis process exiting...")
    
    def analyze_audio_sentiment(self, audio_chunks):
        """Analyze the sentiment of the collected audio chunks."""
        try:
            # Filter for user chunks only
            user_chunks = [chunk for chunk in audio_chunks if chunk['source'] == 'user']
            
            if not user_chunks:
                print("No user audio to analyze")
                return
                
            print(f"\nAnalyzing sentiment for {len(user_chunks)} user audio chunks...")
            
            # Combine user audio chunks for processing
            combined_audio = b''.join([chunk['data'] for chunk in user_chunks])
            
            # Write the audio to a temporary WAV file
            import tempfile
            import wave
            
            # Create a temporary file that gets deleted automatically when closed
            temp_file_path = tempfile.gettempdir() + '/' + next(tempfile._get_candidate_names()) + '.wav'
            
            # Write the audio data to the file
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz sample rate
                wf.writeframes(combined_audio)
            
            # Upload the file for Gemini to analyze
            audio_file = self.sentiment_client.files.upload(
                file=temp_file_path
            )
            
            # Create the prompt
            prompt = "Analyze the sentiment of this audio. Classify it as exactly one of: positive (1), neutral (0), or negative (-1). Return only the number."
            
            # Call the sentiment model
            response = self.sentiment_client.models.generate_content(
                model=self.sentiment_model_name,
                contents=[prompt, audio_file]
            )
            
            # Extract the sentiment score
            sentiment_text = response.text.strip()
            
            # Try to clean up the temp file, but don't cause an error if it fails
            try:
                import os
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Note: Could not delete temporary file {temp_file_path}: {e}")
                # This is not a critical error, so we continue processing
            
            # Parse the response to get the sentiment value
            if "1" in sentiment_text:
                sentiment_value = 1
                sentiment_label = "positive"
            elif "-1" in sentiment_text:
                sentiment_value = -1
                sentiment_label = "negative"
            else:
                sentiment_value = 0
                sentiment_label = "neutral"
                
            # Store the result in shared memory
            result = {
                "timestamp": time.time(),
                "sentiment_value": sentiment_value,
                "sentiment_label": sentiment_label,
                "raw_response": sentiment_text
            }
            self.sentiment_history.append(result)
            
            print(f"Sentiment analysis result: {sentiment_label} ({sentiment_value})")
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            traceback.print_exc()

    def stop(self):
        """Signal the process to exit."""
        self.exit_flag.set()


class AudioLoop:
    def __init__(self, video_mode="none", enable_downstream=False):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SEND_SAMPLE_RATE = 16000
        self.RECEIVE_SAMPLE_RATE = 24000
        self.CHUNK_SIZE = 1024

        self.DEFAULT_MODE = video_mode

        self.api_key = json.load(open(".api_key.json"))["api_key"]

        self.CONFIG = {"response_modalities": ["AUDIO"]}

        self.pya = pyaudio.PyAudio()

        self.voice_client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})
        self.sentiment_model = "models/gemini-2.0-flash-lite"
        self.voice_model = "models/gemini-2.0-flash-exp"
        self.video_mode = video_mode
        self.enable_downstream = enable_downstream  # Flag to enable downstream processing

        self.audio_in_queue = None
        self.out_queue = None
        self.collected_audio_data = []  # List to store collected audio chunks
        
        # For sentiment analysis in a separate process
        self.sentiment_process = None
        self.audio_mp_queue = None
        
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

    async def listen_audio(self):
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, self.CHUNK_SIZE, **kwargs)
                
                # Create user audio chunk
                user_audio_chunk = {
                    "source": "user", 
                    "data": data, 
                    "rate": self.SEND_SAMPLE_RATE,
                    "timestamp": time.time()
                }
                
                # Add to collected data
                self.collected_audio_data.append(user_audio_chunk)
                
                # Send to sentiment analysis process if enabled
                if self.enable_downstream and self.audio_mp_queue:
                    try:
                        # Use non-blocking put with a short timeout
                        self.audio_mp_queue.put(dict(user_audio_chunk), timeout=0.1)
                    except (queue.Full, Exception) as e:
                        # Skip if queue is full or any other error occurs
                        if isinstance(e, Exception) and not isinstance(e, queue.Full):
                            print(f"Error sending to sentiment process: {e}")
                
                # Send to the model
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                print(f"Error in listen_audio: {e}")
                # Continue trying to listen rather than crashing

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
            self.pya.open,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            try:
                bytestream = await self.audio_in_queue.get()
                
                # Create model audio chunk
                model_audio_chunk = {
                    "source": "model", 
                    "data": bytestream, 
                    "rate": self.RECEIVE_SAMPLE_RATE,
                    "timestamp": time.time()
                }
                
                # Add to collected data
                self.collected_audio_data.append(model_audio_chunk)
                
                # Send to sentiment analysis process if enabled
                if self.enable_downstream and self.audio_mp_queue:
                    try:
                        # Use non-blocking put with a short timeout
                        self.audio_mp_queue.put(dict(model_audio_chunk), timeout=0.1)
                    except (queue.Full, Exception) as e:
                        # Skip if queue is full or any other error occurs
                        if isinstance(e, Exception) and not isinstance(e, queue.Full):
                            print(f"Error sending to sentiment process: {e}")
                
                # Play the audio - this should always happen
                await asyncio.to_thread(stream.write, bytestream)
            except Exception as e:
                print(f"Error in play_audio: {e}")
                # Continue trying to play other audio rather than crashing

    async def run(self):
        # Start the sentiment analysis process if enabled
        if self.enable_downstream:
            self.audio_mp_queue = multiprocessing.Queue(maxsize=1000)
            self.sentiment_process = SentimentAnalysisProcess(
                api_key=self.api_key,
                sentiment_model_name=self.sentiment_model,
                audio_queue=self.audio_mp_queue
            )
            self.sentiment_process.start()
            print("Sentiment analysis process started")
            
        try:
            async with (
                self.voice_client.aio.live.connect(model=self.voice_model, config=self.CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                # First start the playback task
                play_audio_task = tg.create_task(self.play_audio())
                
                # Then start the receiving task
                tg.create_task(self.receive_audio())

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
            # Stop the sentiment analysis process if running
            if self.sentiment_process:
                print("Stopping sentiment analysis process...")
                self.sentiment_process.stop()
                self.sentiment_process.join(timeout=5)
                if self.sentiment_process.is_alive():
                    print("Sentiment process did not terminate, forcing termination...")
                    self.sentiment_process.terminate()
                
            # Ensure audio stream is closed if it exists and is open
            if hasattr(self, 'audio_stream') and self.audio_stream and not self.audio_stream.is_stopped():
                 self.audio_stream.stop_stream()
                 self.audio_stream.close()
                 
            # Collect sentiment results if available
            sentiment_results = []
            if self.enable_downstream and self.sentiment_process:
                sentiment_results = list(self.sentiment_process.sentiment_history)
                
            print(f"Collected {len(self.collected_audio_data)} audio chunks and {len(sentiment_results)} sentiment results.")
            
            # Return the results
            return {
                "audio_data": self.collected_audio_data,
                "sentiment_results": sentiment_results
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="none",
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
    results = asyncio.run(main.run())

    # process the collected data
    audio_data = results["audio_data"]
    sentiment_results = results["sentiment_results"] if "sentiment_results" in results else []
    
    if audio_data:
        print(f"Successfully collected audio data. Number of chunks: {len(audio_data)}")
        
        # Print sentiment analysis results if available
        if args.downstream and sentiment_results:
            print("\n===== SENTIMENT ANALYSIS SUMMARY =====")
            print(f"Total sentiment samples: {len(sentiment_results)}")
            
            # Count occurrences of each sentiment
            sentiment_counts = {
                "positive": sum(1 for s in sentiment_results if s["sentiment_value"] == 1),
                "neutral": sum(1 for s in sentiment_results if s["sentiment_value"] == 0),
                "negative": sum(1 for s in sentiment_results if s["sentiment_value"] == -1)
            }
            
            # Print counts
            print(f"Positive responses: {sentiment_counts['positive']}")
            print(f"Neutral responses: {sentiment_counts['neutral']}")
            print(f"Negative responses: {sentiment_counts['negative']}")
            
            # Determine overall sentiment
            if sentiment_counts["positive"] > max(sentiment_counts["neutral"], sentiment_counts["negative"]):
                overall = "positive"
            elif sentiment_counts["negative"] > max(sentiment_counts["neutral"], sentiment_counts["positive"]):
                overall = "negative"
            else:
                overall = "neutral"
                
            print(f"Overall conversation sentiment: {overall}")
            print("======================================")
    else:
        print("No audio data was collected.")
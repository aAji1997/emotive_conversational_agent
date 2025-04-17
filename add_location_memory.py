from memory.memory_utils import MemoryManager

def main():
    # Initialize memory manager
    manager = MemoryManager()
    if not manager.connect():
        print("Failed to connect to Supabase")
        return
    
    # Add the Sacramento/Seattle memory
    user_id = "1cd51eec-8d9c-4f3e-be33-49a2e363073e"  # Samantha's user ID
    content = "User moved from Sacramento to Seattle and has a new apartment"
    importance = 9
    category = "personal_info"
    source = "manual"
    
    memory_id = manager.add_memory(content, importance, category, source, user_id=user_id)
    if memory_id:
        print(f"Successfully added memory with ID: {memory_id}")
    else:
        print("Failed to add memory")

if __name__ == "__main__":
    main()

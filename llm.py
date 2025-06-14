from llama_cpp import Llama  # Replace with actual import

class LLM:
    _instance = None  # Class attribute to hold the single instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:  # Check if an instance already exists
            cls._instance = super(LLM, cls).__new__(cls)  # Create a new instance
            try:
                cls._instance.initialize(*args, **kwargs)  # Initialize the instance
            except Exception as e:
                print(f"Error initializing Llama instance: {e}")
                raise
        return cls._instance  # Return the single instance

    def initialize(self, model_path, n_ctx, n_threads, n_gpu_layers, chat_format):
        try:
            # Initialize the Llama instance
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                chat_format=chat_format,
                # temperature=0,
                verbose=False
            )
            print(f"Llama instance initialized successfully with model_path: {model_path}")
        except Exception as e:
            print(f"Failed to initialize Llama instance: {e}")
            self.llm = None  # Explicitly set to None if initialization fails

    def create_chat_completion(self, *args, **kwargs):
        # Delegate the method call to the Llama instance if it's initialized
        if self.llm is None:
            raise AttributeError("Llama instance is not initialized properly.")
        return self.llm.create_chat_completion(*args, **kwargs)

# Function to get the instance
def get_llm_instance():
    if not LLM._instance:
        # Initialize the LLM instance if it doesn't exist
        LLM(model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Update the path as needed
            n_ctx=2048,
            n_threads=24,
            n_gpu_layers=25,
            # n_parallel=2,
            chat_format="llama-2")
    return LLM._instance

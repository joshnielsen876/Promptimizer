# Promptimizer
A genetic algorithm for evolving prompts. Prompts are decomposed into discrete components and settings which can be used as an index of instructions to become a chromosome. 

I apologize that the codebase is a mess, but it worked well enough for me to finish my PhD! You can read the write-up in the pdf titled Combinatorial Promptimization.
You'll need to set up the config with the benchmark you're testing, which has to be in the same format as gsm8k. If you've got your local model connected (I was using mixtral 8x7B), you should be good to go with main.py.

I was testing it on a classification problem of living kidney donor (LKD) text samples, and the structure works a little bit different in that case. It should work if you use the LKD file versions. 

The goal would be to make it more robust so it works for any kind of prompting case, but that's TBD. I'd also like to get it working for Ollama.


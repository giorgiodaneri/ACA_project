# Mapping Transformers to CiM architectures
Project carried out by Giorgio Daneri and Jacopo Palumbo as part of the Advanced Computer Architectures course of the High Performance Computing Engineering master degree at Politecnico di Milano.
1. Giorgio Daneri (giorgio.daneri@mail.polimi.it)
2. Jacopo Palumbo (jacopo.palumbo@mail.polimi.it)

# Overview
In this article we try different mappings of transformer architectures to compute-in-memory (CIM) systems, highlighting the potential for significant improvements in efficiency and performance. Transformers have revolutionized natural language processing and various other fields with their self-attention mechanisms and parallel processing capabilities. However, their substantial computational and memory demands pose significant challenges for traditional digital computing architectures. CIM architectures offer a promising alternative by integrating memory and computation within the same hardware units, thus minimizing data transfer bottlenecks and enhancing computational throughput.

# Proposed CiM architectures
CiMloop offers several architectures that model the hard-
ware components with different precision levels. While they
weren’t designed specifically for transformers, they can still
achieve good performance and efficiency since the basic oper-
ations are matrix multiplications, which are ubiquitous in the
DNNs world. We focus on three high-fidelity
architectures that have been published in academic journals:
- NeuRRAM, analog RRAM-based chip
- Sinangil, analog SRAM-based architecture
- Colonnade, fully digital SRAM-based architecture

# Explored LLMs
In this analysis we focus on the decoder-only architecture
(GPT-like LLMs) with the objective of finding the best tuple
CiM-LLM to employ LLMs inference on the edge.
- GPT-2 Medium
- Llama2 7B
- Phi-3
- Mistral 7B

# Practical Approach
We used the PyTorch library to load the models and access
the layers, as well as to produce the histograms relative to
the distribution of the input, output and weight values. We
leveraged the remote resources offered by Google Colab and
moved the storage and calculations to the GPU, which offers
acceleration with the CUDA library. Producing the histograms
is fundamental in order to model the power consumption of
the various layers, especially when the workload is mapped on
an analog architecture. The more the values of the matrices
are clustered around the zero, the lower the expected energy
consumption. This is because a value is codified as a tension
when it is converted to analog format by the DAC; the
higher its absolute value, the higher the tension necessary
to represent it in the memory cell. This reasoning does not
directly apply to the Colonnade architecture, which is fully
digital. It is therefore more complex to understand the factors
that contribute to energy consumption in the case of this
architecture. Please read the report for more details. Below is an example of the weight distribution of Llama2 linear layer.

<img src="https://github.com/user-attachments/assets/107f2bd7-1404-4ce8-b250-06ea7ef35d1f" alt="histogram_llama2" width="500"/>

# Mapping the workload on CiMLoop
We used a Cimloop function to perform a full DNN exploration of some workloads corresponding to both older
and state-of-the-art transformers. These include the aforementioned GPT2, Phi-3, Llama-2 and Mistral. We mapped them on
the three architectures described above, among those already
available on Cimloop. On the x-axis we reported the number of
layers of the workload, while on the y-axis we depicted the en-
ergy consumption in femto Joules per MAC operation (1e-15
J/MAC). The input sequence is 10 tokens, thus very short. In
order to produce the histograms, we limited the output length
to a single token so that each layer is visited exactly once. The output is a distribution of energy consumption for each layer of the workload, which can be visualized through a matplotlib bar chart.  
Below you can see the energy consumption of the Phi-3 workload mapped on the NeuRRAM architecture.

![phi3_wan_arch](https://github.com/user-attachments/assets/ecd74ec5-9ad6-41f4-b0e5-1184bc9f98af)




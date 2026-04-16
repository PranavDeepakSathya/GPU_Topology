okay, let me fucking brain dump and you will collect it and make a nice pretty markdown



1. understand the full topology collected on the 4x A100s (including non NVLINK stuff, and the rules of commuication etc etc etc)

2. understand all the communication schemes and why/when they are used. 

3. Collect MORE topologies for thinking against

4.  Is the goal of the project to route alogrithms/communication (communication schemes) or is it to simply answer, hey these are my tensors, they are sitting on these gpus, they need to go to these gpus, and more importantly can algorithms be reduced to a bunch of flows in some nice way

5. is the topology itself reducible (to some extent) to the naive graph problem we currently have? 

6.  what are we doing?

7. Programmatic control: Even if we think of exact routing in the topology, can we program that? or if the hardware takes away some control from us, can we deterministically model what would happen on the hardware if we deployed a particular protocol at the finest granulairty of programmatic control we have? 

8. we have a very naive water flow based modelling, how much of that actually holds with hardware? 
what is the linear programming alogrithm doing? what are the constrians of the water flow? 

https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model read this 


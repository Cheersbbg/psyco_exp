This repo contains my effort to adapt EEG signals to temporal graphs with semantics.
The citation for the original NN is in the subfolder.

**Successful Instances of Aggregation from Sampler**
![successful instances from sampler](visuals/demo_memorymatrix.png)

**From Aggregation of Successful Paths to Generate One Hop Sequence for good representation of Node 10** These are currently cherry-picked from high scoring samples for demostration purpose, longer sequence generation from model is WIP, refer to `find_universal_path_from_subspace` for how to use a symmetric, proximity matrix(created from annotation in ![interface repo](https://github.com/Cheersbbg/Custom-BCI-Experiment-Generator) to compute universal path)

![samples for node 10](https://github.com/Cheersbbg/psyco_exp/blob/main/gooduniversalpath/4768_node_10%5B10%2C%201%2C%201%5D.gif)

![samples for node 10](https://github.com/Cheersbbg/psyco_exp/blob/main/gooduniversalpath/4699_node_10%5B10%2C%2010%2C%209%5D.gif)

![samples for node 10](https://github.com/Cheersbbg/psyco_exp/blob/main/gooduniversalpath/1373_node_10%5B10%2C%206%2C%206%5D.gif)


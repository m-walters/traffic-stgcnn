# traffic-stgcnn
Spatio-Temporal Graph CNN for traffic predictions.  

The traffic data is GPS coordinate readings and times.  
The current code format is my attempt at dividing the geographic map into a grid of cells with size ~1-10 km^2, and producing statistical variables based on traffic data captured in each cell. This grid format would then be the input to a CNN for spatial analysis and subsequently an RNN for the temporal component. However, data was perhaps too sparce, and I recently read some research on Graph CNNs and they seem like a more elegant, if not more effective, solution.

Typical CNN applications are on latticed domains (i.e. images). However, many real-world problems are not on lattices, and yet exhibit a spatial relationship. Graph-CNNs are a more general type of CNN where positional relationships are encoded as vertices in a graph. I say more general because a regular pxp image is equivalently a graph with connections between neighbouring pixels.  

One may quickly see how this relates to traffic prediction. Segments of roads would be graph nodes, and vertices occur where roads are connected. Each node would have one or more statistical variables, such as average velocity and number of cars.  

Dataset was collected from Sep 30 2015 -- Oct 31 2015

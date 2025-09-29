KMeans is an unsupervised learning method (unlabeled data) that clusters data set into k diffirent clusters. 
each sample is assigned to the cluster with the nearest mean, end then the mean (centroids) and clusters are updated 
during an iterative optimization process.
### procedure
- update cluster labels: assign points to the nearest cluster center (centroid)
- update cluster centers: set center to the mean of each cluster
  
Initially, we randomly initialize k centroids from the original data points. We then use the Euclidean distance formula to calculate the distance between the data points and assign each point to the cluster of the nearest centroid. After that, we update the coordinates of the centroids by taking the mean of all the samples in each cluster. This process is repeated continuously until no data point changes its cluster anymore, or until the maximum number of iterations is reached.

<img width="1202" height="800" alt="Figure_-1" src="https://github.com/user-attachments/assets/fde41ee7-d174-485d-821b-9e9bf5cd4baa" />

<img width="1202" height="800" alt="Figure_0" src="https://github.com/user-attachments/assets/9dd1037b-7b95-4e69-b1a8-8ee1738aa416" />

<img width="1536" height="754" alt="Figure_1" src="https://github.com/user-attachments/assets/26bad0e6-7b73-4930-9d7e-3c4f719edb80" />

<img width="1536" height="754" alt="Figure_2" src="https://github.com/user-attachments/assets/e8a5f598-9b76-4ac9-9aa7-561fde4d34a7" />

<img width="1536" height="754" alt="Figure_3" src="https://github.com/user-attachments/assets/ce09e186-bf3c-4b37-b37c-e8736b8678d2" />

<img width="1536" height="754" alt="Figure_4" src="https://github.com/user-attachments/assets/6baeae6b-d7fd-4bf6-b22a-8119b9d6b69c" />

<img width="1536" height="754" alt="Figure_5" src="https://github.com/user-attachments/assets/78af33a9-db00-4d15-b07e-15d0663e3dc3" />

<img width="1536" height="754" alt="Figure_6" src="https://github.com/user-attachments/assets/beae1b88-da49-418b-8585-5aaee065500b" />

<img width="1536" height="754" alt="Figure_7" src="https://github.com/user-attachments/assets/6cceb014-3c08-4a7b-90a6-c834ead55789" />

<img width="1536" height="754" alt="Figure_8" src="https://github.com/user-attachments/assets/ad3c3f23-e6cf-4790-82ab-52b65b71a7d6" />

<img width="1536" height="754" alt="Figure_9" src="https://github.com/user-attachments/assets/006e117e-5d5f-4887-a7af-4f57eb2a9808" />

<img width="1536" height="754" alt="Figure_10" src="https://github.com/user-attachments/assets/44731a41-4c5c-4da3-a4a0-77f55cc62b82" />

<img width="1536" height="754" alt="Figure_11" src="https://github.com/user-attachments/assets/b4226549-b584-477b-8c81-659d23ca79c2" />

<img width="1536" height="754" alt="Figure_12" src="https://github.com/user-attachments/assets/fa7b7605-92cf-47d1-be38-df0deb633149" />





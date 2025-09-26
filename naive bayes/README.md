Naive Bayes classifier is a probabilistic classifier basesd on applying Bayes's theorem with strong independence assumptions between the features.
# Bayes's theorem
<img width="973" height="406" alt="{38865CAD-0696-4777-A714-ED274AC50550}" src="https://github.com/user-attachments/assets/b4bf964e-cf4c-41ee-a28c-25743507d0ba"/>


Assume that the features must be independent of each other; then we have:
<img width="955" height="335" alt="{94276964-4C0C-4040-9687-B792FF57F6F7}" src="https://github.com/user-attachments/assets/92260d9d-bf76-4a3c-ad4b-ec02b887c874" />

# If so, how does the model work?
It chooses the class with the highest posterior probability.
<img width="887" height="355" alt="{D2BC9BA0-6A88-4EFA-BCF9-41E43992894E}" src="https://github.com/user-attachments/assets/50131536-7e5a-4f04-8dda-1dedc7b47c92" />

We will consider the posterior probability of a sample based on all the available data (inputs). Since all posterior probabilities are divided by 
P(X), we can omit it, as we only need to select the class with the highest numerHowever, the probability values can be very small, and multiplying them may make the posterior probability extremely tiny, causing the computer to round the result to 0. Therefore, we apply the logarithm function, which not only addresses the issue of very small values but also allows us to convert the product into a sum for easier computation.ator.







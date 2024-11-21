Identifying deep voice samples generated by AI has been challanging tasks since the introduction of large scale generative models that can mimic human voice and accent. In this regard, building frameworks to differentiate natural voice samples from AI generated audio is an essential task for an AI safety.

During Mid-Term project of Machine Learning ZoomCamp I have build and deployed a system that can identify artificial audio clip from natural by analyzing frequency response of the given audio and applying Support Vector Machine and Boosting algorithms to classify sample.

Performance of tested algorithms:
- Ridge Classifier: 
    - 'accuracy_score': 0.890
    - 'precision_score': 0.893
    - 'recall_score': 0.891
    - 'f1_score': 0.892
- Support Vector Classifier:
    - 'accuracy_score': 0.984
    - 'precision_score': 0.982
    - 'recall_score': 0.986
    - 'f1_score': 0.984
- Gradient Boosting Classifier:
    - 'accuracy_score': 0.988
    - 'precision_score': 0.989
    - 'recall_score': 0.987
    - 'f1_score': 0.988

How to run the .ipynb files, you will need to follow the steps below:
- build virtual environment via pipenv
- install the required libraries listed in requirements.txt
- pipenv install -r requirements.txt && pipenv lock && pipenv shell
- activate the environment and run ipynb notebooks under given environment

How to deploy the model with Docker container:
- docker build --force-rm -t deepvoice -f Dockerfile .
- docker run -it --rm -p 9696:9696 deepvoice
- change visibility of port 9696 to public under 'PORTS' in VS code
- change 'url' in /src/client.py to the Forwarded Address connected to port 9696
- test pipeline from client side by running cd /src/ && python -m client

References:
- Kolmogorov-Smirnov Test on significance of different spectral variables https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
- Wilcoxon rank-sum statistic to test that two sets of measurements are drawn from the same distribution https://www.stat.auckland.ac.nz/~wild/ChanceEnc/Ch10.wilcoxon.pdf


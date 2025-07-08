### Prerequisites
- [Conda](https://conda.io/docs/)
- [LibMamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)

1. Build and activate conda environment:
   ```shell
   conda env create -f environment.yml && conda activate CanExClasif
   ```
   
2. Update the environment when adding package to environment.yml 
   ```shell
   conda env update -f environment.yml
   ```
   
3. Install pre-commit hooks
   ```shell
   pre-commit install
   ```
   
4. [Optional] Remove conda env
   ```shell
   conda deactivate && conda remove -n data-iotcore-lib --all
   ```

5. Make sure to drop the following videos in the `data/raw` folder:
   - `GS010012.mp4`
   - `GS010016.mp4`

### Raw Data   
[Canex Onedrive](https://canextechno-my.sharepoint.com/personal/olivier_beaulieu_canex_tech/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Folivier%5Fbeaulieu%5Fcanex%5Ftech%2FDocuments%2FTest%20Technique%20Data&ga=1)

### Task 1: Multiple Pipe Segment Detection

You are provided with multiple 360-degree videos from real sewer inspections. Each video contains two pipe segments,
separated by a manhole. Your task is to design and implement an algorithm that can automatically detect and locate the middle
manhole within each video.

Your submission should include:
- Approach:
Describe your end-to-end methodology, including data preprocessing, model selection, and feature engineering.
Explain any assumptions you make about the data or the problem.
- Algorithm/Model:
- Provide code (in Python, using any open-source libraries) that demonstrates your detection pipeline. Clearly comment your code and highlight key decisions.
- Evaluation: Propose metrics to evaluate your solutionʼs performance. If possible, include a brief analysis of strengths, weaknesses, and potential improvements.
- Deployment Considerations: Briefly discuss how your solution could be deployed in a real-world setting (e.g., edge device, cloud).
- Mention any challenges you foresee in scaling or maintaining your approach.

### Task 2: Tap Detection Problem
You are provided with a sample 360-degree video from a real sewer inspection. Your task is to design and implement an
algorithm that can automatically detect the presence and location of a Tap within the pipe. An example image of a Tap (lateral
pipe connection) will also be provided for reference.
Your submission should include:
- Approach:
Describe your end-to-end methodology, including data preprocessing, model selection, and feature engineering.
Explain any assumptions you make about the data or the problem.
- Algorithm/Model:
Provide code (in Python, using any open-source libraries) that demonstrates your detection pipeline.
Clearly comment your code and highlight key decisions.
- Evaluation:
Propose metrics to evaluate your solutionʼs performance.
If possible, include a brief analysis of strengths, weaknesses, and potential improvements.
- Deployment Considerations:
Briefly discuss how your solution could be deployed in a real-world setting (e.g., edge device, cloud).
Mention any challenges you foresee in scaling or maintaining your approach.

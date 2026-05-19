from setuptools import setup, find_packages

setup(
    name='dynasty-ai',
    version='1.0.0',
    description='An AI engine to assist with decision-making for dynasty fantasy football leagues',
    author='Sandeep Tiwari',
    author_email='tiwaristiwari97@gmail.com',
    packages=find_packages(),
    install_requires=[
        'nfl_data_py>=0.3.3',
        'cfbd>=1.0.0',
        'requests>=2.31.0',
        # 'pandas>=2.0.0',
        'pandas',
        'numpy>=1.24.0',
        'sqlalchemy>=2.0.0',
        'lightgbm>=4.0.0',
        'scikit-learn>=1.3.0',
        'shap>=0.43.0',
        'langchain>=0.2.0',
        'langchain-anthropic>=0.1.0',
        'langchain-chroma>=0.1.0',
        'langgraph>=0.1.0',
        'tavily-python>=0.3.0',
        'streamlit>=1.35.0',
        'jupyter>=1.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'mlflow>=2.12.0',
        'rapidfuzz'
    ],
)
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="anime_recommender_system",
    version="0.1.0",
    author="Jayesh",
    packages=find_packages(),
    install_requires=requirements,
    description="A simple anime recommender system using collaborative filtering.",
)
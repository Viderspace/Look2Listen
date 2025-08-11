# """
# Setup configuration for AV-Speech Enhancement package
# """
# from setuptools import setup, find_packages
# from pathlib import Path
#
# # Read the README for long description
# readme_path = Path(__file__).parent / "README.md"
# long_description = ""
# if readme_path.exists():
#     long_description = readme_path.read_text(encoding="utf-8")
#
# # Read requirements
# requirements_path = Path(__file__).parent / "requirements-colab.txt"
# requirements = []
# if requirements_path.exists():
#     requirements = [
#             line.strip()
#             for line in requirements_path.read_text().splitlines()
#             if line.strip() and not line.startswith("#")
#     ]
#
# setup(
#         # Package metadata
#         name="av-speech-noise-reduction",
#         version="0.1.0",
#         author="Your Name",
#         author_email="your.email@example.com",
#         description="Visual-guided speech enhancement using lip movements and audio-visual fusion",
#         long_description=long_description,
#         long_description_content_type="text/markdown",
#         url="https://github.com/yourusername/av-speech-enhancement",
#
#         # Package configuration
#         packages=find_packages(exclude=["app", "app.*", "tests", "tests.*", "docs"]),
#         package_dir={"avspeech": "avspeech"},
#
#         # Include non-Python files (if needed)
#         package_data={
#                 "avspeech": [
#                         "utils/*.json",  # If you have config files
#                         "model/*.pt",  # If you include model weights
#                         "model/*.pth",
#                 ],
#         },
#         include_package_data=True,
#
#         # Dependencies
#         install_requires=requirements,
#
#         # Optional dependencies for specific use cases
#         extras_require={
#                 "dev"     : [
#                         "pytest>=6.0",
#                         "pytest-cov",
#                         "black",
#                         "flake8",
#                         "ipython",
#                 ],
#                 "app"     : [
#                         "gradio>=4.0.0",
#                 ],
#                 "training": [
#                         "tensorboard",
#                         "wandb",  # if you use wandb for experiment tracking
#                 ],
#         },
#
#         # Python version requirement
#         python_requires=">=3.8",
#
#         # Entry points for command-line scripts (optional)
#         entry_points={
#                 "console_scripts": [
#                         # Add CLI commands here if needed
#                         # "avspeech-process=avspeech.cli:main",
#                         # "avspeech-train=avspeech.training.train:main",
#                 ],
#         },
#
#         # Classifiers for PyPI (optional but good for documentation)
#         classifiers=[
#                 "Development Status :: 3 - Alpha",
#                 "Intended Audience :: Developers",
#                 "Intended Audience :: Science/Research",
#                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
#                 "Topic :: Multimedia :: Sound/Audio :: Speech",
#                 "Topic :: Multimedia :: Video",
#                 "License :: OSI Approved :: MIT License",  # Update with your license
#                 "Programming Language :: Python :: 3",
#                 "Programming Language :: Python :: 3.8",
#                 "Programming Language :: Python :: 3.9",
#                 "Programming Language :: Python :: 3.10",
#                 "Programming Language :: Python :: 3.11",
#         ],
#
#         # Keywords for search
#         keywords="speech-enhancement audio-visual deep-learning lip-reading noise-reduction",
# )
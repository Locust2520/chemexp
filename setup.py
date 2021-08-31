from distutils.core import setup

readme = ""
try:
  with open("README.md", 'r') as f:
    readme = f.read()
except:
  pass

setup(
  name = "chemexp",
  packages = ["chemexp"],
  version = "0.2",
  description = "An explainer for Chemprop based on LIME and PathExplain.",
  long_description = readme,
  long_description_content_type = "text/markdown",
  author = "Fabien BERNIER",
  author_email = "fabien.bernier@telecomnancy.net",
  url = "https://github.com/Locust2520/chemexp",
  keywords = [
    "explainer",
    "explanation",
    "visualization",
    "chemprop",
    "molecule",
    "model",
    "deep learning",
    "graph"
  ],
  classifiers = [],
)

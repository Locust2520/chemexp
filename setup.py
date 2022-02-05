import pathlib
from distutils.core import setup


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
  name = "chemexp",
  packages = ["chemexp"],
  package_data={"chemexp": ["static/*", "templates/*"]},
  version = "0.2.4",
  description = "An explainer for Chemprop based on LIME and PathExplain.",
  long_description = README,
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
  classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8"
  ],
)

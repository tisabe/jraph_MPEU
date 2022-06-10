import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

print("HERE!!!!", setuptools.find_packages("./src"))

setuptools.setup(
    name="jraph_MPEU_all",
    version="0.9",
    author="test",
    author_email="bechtelt@physik.hu-berlin.de",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages = setuptools.find_packages(".")
    #packages=['src/jraph_MPEU', 'src/jraph_MPEU_configs']
)

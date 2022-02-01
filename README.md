# Enabling autonomous driving through Zenoh Flow

Project largely inspired by [Pylot](https://github.com/erdos-project/pylot).

Thank you the Pylot team for your contribution to open autonomous driving.

## Getting Started

To make this project work, you will need to:
- clone and install [pylot](https://github.com/erdos-project/pylot) in the parent directory using the install script.
- clone and install [zenoh-flow-python](https://github.com/atolab/zenoh-flow-python) in the parent directory.
- Install the requirements with:

```bash
pip install -r requirements.txt
```

> It's possible that cv2 might not have all its dependencies. You should try installing with conda with:

```bash
conda install -c menpo opencv
```
- Get an image

```bash
cd data
wget https://www.smartrippers.com/files/2017-11/panneau-feu-usa2.jpg
cd ..
```

- Run the following command for running operator on standalone:
```bash
python tests/tests_integrations.py
```
- Run the following command to run with Zenoh-flow:

```bash
../path/to/runtime -r foo -g py-pipeline.yml -l loader-config.yml
```
# Berkeley Humanoid Lite

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)
[![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-orange.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

**[Website](http://lite.berkeley-humanoid.org/)** | **[arXiv](https://arxiv.org/abs/2504.17249)** | **[Paper](https://lite.berkeley-humanoid.org/static/paper/demonstrating-berkeley-humanoid-lite.pdf)** | **[Video](https://youtu.be/dIdJGkMDFl4?si=SRD7HhQQbhM3JCRA)** | **[Documentation](https://berkeley-humanoid-lite.gitbook.io/berkeley-humanoid-lite-docs)** | **[Releases](https://berkeley-humanoid-lite.gitbook.io/docs/releases)**


Berkeley Humanoid Lite is an open-source, sub-$5,000 humanoid robot featuring modular 3D-printed gearboxes and widely available components, designed to democratize and advance humanoid robotics research.

This project is built on the values of open-source, accessibility, and customization, and it's continuously evolving. We welcome your feedback, issues, and pull requests on GitHub or joining our Discord.

## Overview

This repository is the workspace for the Berkeley Humanoid Lite project that contains everything we need, including policy training, sim2sim validation, real-world deployment, motion capture, and teleoperated manipulation controls.

Functionalities are organized into several submodules. We arrange the directory structure following the Isaac Lab convention, where each submodule can be installed as an extension:

- `source/berkeley_humanoid_lite/` contains the IsaacLab environment and task definitions.

- `source/berkeley_humanoid_lite_assets/` contains robot descriptions (URDF, MJCF, and USD) and the script to export these description files from Onshape project.

- `source/berkeley_humanoid_lite_lowlevel/` contains the lowlevel code running on the real robot. Only contents inside this folder are required to deploy to the real robot.

Except a few edge cases, all the commands should be invoked from the root directory of this repository. The entry points of different flows are collected in the `scripts/` directory.


## Getting Started

Please refer to our [Documentation](https://berkeley-humanoid-lite.gitbook.io/docs) to get started with software and hardware setup.

The latest release of CAD model and 3D print files can be accessed from the [Release](https://berkeley-humanoid-lite.gitbook.io/docs/releases) page.


## Contributing

We wholeheartedly welcome contributions from the community to make this robot platform more mature and useful for everyone. We appreciate any kind of contributions, including bug reports, feature requests, or code contributions.

Also, please reach out to us to tell us about your projects and how you are using this robot platform. We would love to feature your work on our website and social media.

## License

The code in this repository is licensed under [MIT License](https://opensource.org/license/mit). See the [LICENSE](LICENSE) file for details.

Other assets are under [Creative Commons Attribution-ShareAlike 4.0 International <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt="">](https://creativecommons.org/licenses/by-sa/4.0).


## Citation

If you find this code useful, we would appreciate it if you would cite our paper:

```
@article{chi2025demonstrating,
  title={Demonstrating Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot},
  author={Yufeng Chi and Qiayuan Liao and Junfeng Long and Xiaoyu Huang and Sophia Shao and Borivoje Nikolic and Zhongyu Li and Koushil Sreenath},
  year={2025},
  eprint={2504.17249},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2504.17249}, 
}
```

# <img src="https://user-images.githubusercontent.com/23240128/233209820-821715e0-07e6-4dbc-8133-d915a7ea06b7.png" width="40" style="vertical-align: middle;"> MyoWeb: Online Musculoskeletal Simulation with RL Control

An interactive, web-based showcase for musculoskeletal simulations built on top of [myosuite_demo](https://github.com/MyoHub/myosuite_demo).



## Overview

This project provides a hands-on demonstration of contact-rich musculoskeletal control using MuJoCo rendered in the browser via WebAssembly. This demo is heavily inspired by and built on top of the following projects:

* [**myosuite\_demo**](https://github.com/MyoHub/myosuite_demo): Example scenes and utilities that tie everything together.
* [**MyoSuite**](https://github.com/MyoHub/myoSuite): Core musculoskeletal modeling and simulation framework.
* [**mujoco\_wasm**](https://github.com/zalo/mujoco_wasm): WebAssembly build of MuJoCo for in-browser physics simulations.


## Features

* **Real-time, browser-based simulation** of human arm, hand, elbow, and finger models.
* **Interactive controls**: Pause, play, reload, reset simulations; manipulate actuators; switch scenes.
* **Reinforcement Learning (RL) integration**: Load and test baseline RL policies via ONNX models.
* **Extensible**: Add custom MuJoCo XML/MJB scenes or integrate new RL policies.


## Learn More

* **DeepWiki**: [https://deepwiki.com/ttktjmt/myosuite\_demo](https://deepwiki.com/ttktjmt/myosuite_demo)
* **MyoSuite discussions**: [https://github.com/MyoHub/myosuite/discussions/292](https://github.com/MyoHub/myosuite/discussions/292)

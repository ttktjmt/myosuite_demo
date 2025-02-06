## Why .mjb is used instead of .xml?

The [myoarm_bionic_bimanual.xml](https://github.com/MyoHub/myosuite/blob/main/myosuite/envs/myo/assets/arm/myoarm_bionic_bimanual.xml) file depends on many other files—recursively including additional XMLs, meshes (.stl), textures (.png), and more.  
When using the XML directly, you must ensure that all the relative paths are correct and that every dependency is available in the expected location. This is not only tedious but also prone to errors.

To avoid these issues, it’s recommended to create a single binary model file (.mjb) that bundles all the dependencies together. This way, you only need to manage one file, and the asset paths inside the model are self‐contained.

For example, in the XML you might see:

```xml
  ...

  <include file="../../../../simhive/YCB_sim/includes/defaults_ycb.xml"/>
  <include file="../../../../simhive/YCB_sim/includes/assets_009_gelatin_box.xml"/>

  <include file="../../../../simhive/myo_sim/arm/assets/myoarm_assets.xml"/>
  <include file="../../../../simhive/myo_sim/scene/myosuite_scene.xml"/>

  <include file='../../../../simhive/MPL_sim/assets/left_arm_assets.xml'/>
  <include file='../../../../simhive/MPL_sim/assets/handL_assets.xml'/>
  
  ...
```


## How to convert .xml files into a single .mjb file?

You can convert your model into one .mjb file using a python script like below. This process loads your XML model (with all its dependencies) and saves a binary version containing all the embedded assets.

```py
import mujoco

model = mujoco.MjModel.from_xml_path("your_model.xml")
mujoco.mj_saveModel(model, "your_model.mjb", None)

print("Model successfully saved!!")
```

Be sure to use the same version of mujoco library as the one used in this project.

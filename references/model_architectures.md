#### Key model architectures and terms:
- ResNet50_1x1: adding conv1x1 to original ResNet50 used by sen12ms (supervised training) 
- ResNet50: original ResNet50 used by sen12ms (supervised training) 
- Moco: original ResNet50 initialized the weight by Moco backbone (transfer learning) 
- Moco_1x1: ResNet50_1x1 initialized the weight by Moco backbone and input module  (transfer learning) 
- Moco_1x1random: ResNet50_1x1 randomly the weight by Moco backbone and input module  (transfer learning) 

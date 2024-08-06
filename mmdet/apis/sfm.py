from modeler import SfM

sfm = SfM('/media/bys2058/Storage/central bridge with drone_7_20/', False, '/media/bys2058/Storage/central bridge with drone_7_20/DJI_0057.MOV', 48)
sfm.find_structure_from_motion()
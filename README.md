# Adaptive-angle-selection-for-defect-detection-on-CT-data
This repository supports a paper submission to the 'Topic Collection of the Journal of Nondestructive Evaluation'. It is designed to facilitate collaboration by sharing code and data pertinent to our research.
- Shepp-Logan: This section contains code and data related to the Shepp-Logan scenario, a more intricate and multi-material case.
  - Data_set_code: It consists of four scripts:
     - SheppLogan_In_External_clean.py: This is the first Shepp-Logan dataset for training derived from Virginia's network. The illustration is ![First_Shepp_Logan_dataset](Shepp_Logan/Shepp_data_In_External_clean.pdf)
     - SheppLogan_Internal_clean.py: This is the second Shepp-Logan dataset for training, again based on Virginia's network. The illustration is ![Second_Shepp_Logan_dataset](Shepp_Logan/Shepp_data_Internal_clean.pdf) (Note: A separate dataset was created by removing the external part of the Shepp-Logan, based on findings from a previous project that indicated the external part influenced the results.)
     - SheppLogan_In_External_defect.py: This represents the first defect dataset associated with SheppLogan_In_External_clean.py. It's used for training on Tianyuan's network and testing on Virginia's network. The illustration is ![First_defect_dataset](Shepp_data_In_External_defect.pdf)
     - SheppLogan_Internal_defect.py: This represents the second defect dataset associated with SheppLogan_In_External_clean.py. It's used for training on Tianyuan's network and testing on Virginia's network. The illustration is ![Second_defect_dataset](Shepp_data_Internal_defect.pdf)
       


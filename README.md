# Adaptive-angle-selection-for-defect-detection-on-CT-data
This project aims to submit a paper to 'Topic collection of the Journal of Nondestructive Evaluation'. This repository is used to share code and data to have a better collobaration.

- Shepp-Logan: Code and data related to Shepp-Logan scenatio, which is more complex and multi-material.
  - Data_set_code: It consists of four scripts:
     - SheppLogan_In_External_clean.py: The first shepp logan data set for the training from Virginia's network and the illustration is ![First_Shepp_Logan_dataset](Shepp_Logan/Shepp_data_In_External_clean.pdf)
     - SheppLogan_Internal_clean.py: The second shepp logan data set for the training from Virginia's network and the illustration is ![Second_Shepp_Logan_dataset](Shepp_Logan/Shepp_data_Internal_clean.pdf) (The reason to have two different data sets: from previous project the external part of Shepp-Logan will influence the results, so I remove the external part for the second data set.)
     - SheppLogan_In_External_defect.py: The first defect data set corresponding to SheppLogan_In_External_clean.py, which is used to train on Tianyuan's network and as the test for Virginia's network. The illustration is ![First_defect_dataset](Shepp_data_In_External_defect.pdf)
     - SheppLogan_Internal_defect.py: The second defect data set corresponding to SheppLogan_Internal_clean.py, which is used to train on Tianyuan's network and as the test for Virginia's network. The illustration is ![Second_defect_dataset](Shepp_data_Internal_defect.pdf)
       


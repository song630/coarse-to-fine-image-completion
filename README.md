# coarse-to-fine-image-completion
## implementation:
* user-specified rectangular region to be completed
* build Gaussian pyramid
* randomly fill the hole (image at the top level)
* employ coherence measurement to find ANN, update color during several iterations
* get down to the next level and propagate the locations of ANN
## platform:
* opencv
## references:
* Denis Simakov, Yaron Caspi, Eli Shechtman, Michal Irani. Summarizing Visual Data Using Bidirectional Similarity. IEEE CVPR, 2008.
* Yonatan Wexler; Eli Shechtman; Michal Irani. Space-Time Completion of Video. IEEE TPAMI, 2007.

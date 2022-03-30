# X-Ray CT scan manipulation of citrus fruits and their individual tissues

## General information

### Author

- **Erik J. Amézquita**, _Michigan State University_

### To whom correspondence should be addressed:

**Erik J. Amézquita**
428 S Shaw Ln
Engineering Building Rm 1515
East Lansing, MI 48824
USA
amezqui3@msu.edu

### Date and geographic location of data collection

Data collected in December 2018 at Michigan State University. Citrus samples provided by the [_Givaudan Citrus Variety Collection_](https://citrusvariety.ucr.edu/) at  University of California Riverside.

### Keywords

- Citrus
- Oil glands
- X-ray computed tomography (CT)
- Plant morphology
- 3D morphology

### Related content

- Amézquita EJ, Quigley M, Ophelders T, Seymour D, Munch E, Chitwood DH "The shape of aroma: measuring and modeling citrus oil gland distribution" _forthcoming_.

- Amézquita EJ, Quigley M, Ophelders T, Seymour D, Munch E, Chitwood DH "X-Ray CT scans of citrus fruits and their individual tissues." Dryad Repository


### License

MIT License

Copyright (c) 2022 Erik  Amézquita

See `LICENSE` for additional details

### Acknowledgements

Work supported by the USDA National Institute of Food and Agriculture, Michigan State University AgBioResearch, and the National Science Foundation through grants CCF-1907591, CCF-2106578, and CCF-2142713.

=========

## Data and file overview

### Overview

We selected 51 different citrus varieties with diverse morphologies and geographical origins for our analysis. 166 different individuals in total were sent for scanning at Michigan State University in December 2018 (c.f. `CRC_citrus_scanned.csv`). These 166 samples were arranged into 63 raw scans, one scan per citrus variety containing all the replicates. An exception were pummelos and citrons, where each sample was individually scanned due to the fruit size. The scans were produced using the North Star Imaging X3000 system and the included efX software, with 720 projections per scan, at 3 frames per second and with 3 frames averaged per projection. The data was obtained in continuous mode. The X‐ray source was set to a current of 70 μA, voltage ranging from 70 kV to 90 kV, and focal spot sizes ranging from 4.9 to 6.3μm. The 3D reconstruction of the citrus was computed with the efX-CT software, obtaining a final voxel size ranging from 18.6 to 110.1 μm for different scans (c.f. `citrus_preprocessing.pdf` and `CRC_citrus_scan_technique`)

There is a special focus on exploring oil gland shape and distribution. We also explore the  overall fruit shape. Overall fruit is approximated with the best algebraic-fit ellipsoid, adapted from [Li and Griffiths (2004)](https://doi.org/10.1109/GMAP.2004.1290055). This produces a 10-dimensional vector that algebraically defines an ellipsoid. See [Panou et al. (2020)](https://doi.org/10.1515/jogs-2020-0105) on how to convert this vector into geometric parameters, like semi-axes lengths, center of the ellipsoid, and rotation angles.

### File description

The whole dataset is split into four (4) folders. Read the individual README files within each folder for more details, and see Amézquita _et al._ (forthcoming) for more information.

- `allometry -> data`: CSV files with volumetric data for each citrus fruit. Each file is a single row with 8 entries, corresponding to:
    - citrus_id
    - label (L00, L01, L02)
    - Number of voxels contained by the whole fruit, exocarp, endocarp, rind, spine, mesocarp, and oil gland tissues respectively. Keep in mind that the volume represented by voxels in each scan varies according to the resolution. Resolution details can be found in the `citrus_voxel_size.csv` file
    - Number of separate oil glands

-  `comps`: Collection of 8-bit TIFF files for individual fruits. Air and noise was removed. For most of the species, 3 samples are provided (see Overview). These samples are labeled `L00`, `L01`, `L02` accordingly. Density **was not** normalized across scans.

- `glands`: Collection of figures and plots for visual assessment of the data. See the `oil` folder description below for more details. For each fruit, the following files are present.
    - `ell_geocentric_fit.jpg`: Plot of the best fit ellipsoid. 
    - `glands.jpg`: 2D projection of the oil glands
    - `diff_semiaxes`: Distribution of differences between semiaxes for all the oil gland MVEEs.
    - `eli_glands_1000.jpg`: Plot of MVEEs for some oil glands. In magenta is the 1000th oil gland. In blue are the 12 closest oil glands. In gray are the next 18 closest glands. Distance is euclidean. Keep in mind that matplotlib 3D plots usually distort axes ratio and depth perception.
    - `gland_semiaxes.jpg`: Distribution of the semiaxes lengths for all the oil gland MVEEs.
    - `gland_sphericity.jpg`: Distribution of various sphericity indices for the oil gland MVEEs.
    - `helices_phglands.jpg`: Distribution of oil gland MVEE phenotypes across fruit height. 
        - 1st row: We consider slicing the fruit 24 times with cuts parallel to its equator. We then consider the average phenotype values for the oil glands contained within sequential slices. In this case, we consider the average MVEE surface area and volume of oil glands within slices. Additionally, we consider average MVEE angle, which is the angle formed between the major semiaxes direction and the equatorial plane. Finally, we keep track of the percentage of glands within each pair of slices.
        - 2nd row: Same as above, except we now slice the best fit fruit ellipsoid.
    - `helices_sphericity.jpg`: Same idea as before, this time considering average sphericity indices of MVEEs between slices.
    - `hlices_sphericity.jpg`: See above.
    - `knn12_distance_ell_center.jpg`: Consider the center of all the gland MVEEs. Distribution of euclidean distance to the nearest neighbor, 2nd nearest neighbor, ..., 12th nearest neighbor.
    - `knn12_distance_vox_center.jpg`: Same as above, but now the centers refer to the center of mass of the voxels that make up individual oil glands.
    - Additional files related to directional statistics, like spherical kernel density estimation, may be found in some fruits.

- `oil`: Collection of TIFF files containing all the segmented oil glands for each citrus. Additional numbers are extracted related to the oil glands. For each fruit, the following files are present. See the overview section above for more details.
    - `glands.csv`: XYZ coordinates for the centers of mass of voxels for each separate oil gland.
    - `amvee.csv`: Semiaxes lengths for MVEEs for each separate oil gland. Lengths are sorted. Lenghts are in voxel size, so they **must** be rescaled with the appropriate resolution value.
    - `cmvee.csv`: Center coordinates for each MVEEs _with respect to_ individual oil gland slicing
    - `dmvee.csv`: Displacement of the oil gland slicing _with respect to_ the whole fruit.
    
     To obtain the center coordinates for each MVEEs _with respect to_ the whole fruit, simply add the values cmvee + dmvee
     
    - `rmvee.csv`: 3x3 rotation matrices for each MVEE for their correct positioning
    - `ell_m_ell.csv`: 6x3 matrix containing geometric parameters of the best algebraic ellipsoid fit for the whole fruit. The fit was based from the pointcloud consisting of the MVEE centers of oil glands.
        - 1st row: center of the ellipsoid
        - 2nd row: semiaxes lengths
        - 3rd-5th rows: rotation matrix
        - 6th matrix: rotation angles
    - `ell_v_ell.csv`: Same ellipsoid as above, but now described as a 10-dim vector
    - `vox_m_ell.csv`: Similar 6x3 matrix with paramters describing the best algebraic ellipsoid fit for the whole fruit. The fit was based from the pointcloud consisting of the center of mass of voxels of separate oil glands.
    - `vox_v_ell.csv`: Analogue
    - `glands_phenotypes.csv`: We consider slicing the fruit 24 times with cuts parallel to its equator. We then consider the average phenotype values for the oil glands contained within sequential slices. In this case, we consider the average MVEE sphericity indices, as well, as the standard deviation and variance within each slice.
    - `ellipsoid_phenotypes.csv`: Same as above, except that we slice the best fit overall ellipsoid instead.
    - `knn_ell_distances.csv`: 100x3 matrix. Consider the center of all the gland MVEEs. Average euclidean distance to the nearest neighbor, 2nd nearest neighbor, ..., 100th nearest neighbor. Also standard deviation and variance for each kth nearest neighbor.
    - `knn_vox_distances.csv`: Same as above, but now we consider the centers of mass of voxels for separate oil glands.


- `tissue`: Separate TIFF files for separate tissues. All images have the same shape as the original whole fruit image.
    - `vh_alignment.csv`: 3x3 rotation matrix so that the spine stands upright. Used to align all the fruits and tissues.

- `citrus_voxel_size.csv`: CSV with resolution of the scans.

- `citrus.xslx`: Metadata provided by the CRC

- `LICENSE`: raw text file with CC0 License details

- `LICENSE_summary`: raw text with CC0 human-readable summary.

- `README.md`: This file. Markdown format. Raw text.

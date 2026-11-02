# The Image-Based Redshift Estimate Challenge:

This repository describes the challenge and dataset for estimating redshifts based on images.

## Context
A critical task for most extragalactic, astronomical studies is to estimate the redshift to a galaxy. Redshifts are typically measured using spectroscopic observations. However, it is not possible to obtain spectroscopy for every galaxy due to the large observation time requirement, particularly for the faintest galaxies. Instead, there has been substantial effort to develop methodology to estimate redshifts based on broadband imaging of galaxies, which is much easier to obtain for large samples. We will call redshifts measured from broadband imaging “photometric redshifts” or “photo-zs”, and those from spectroscopy “spectroscopic redshifts” or “spec-zs”. 

Traditionally, photo-z methods rely on integrated photometric quantities. Using either apertures (e.g., ellipses) or model fits to galaxy images, total magnitudes and colors are extracted (see, e.g., the photutils tool for examples of how this is done). There are correlations between the total magnitudes, colors, and redshifts, albeit with large uncertainties and, sometimes, strong degeneracies. These methods are generally not developed with the faintest or lowest mass galaxies in mind.

Methodology to produce precise and accurate photometric redshifts for low mass galaxies is essential to many areas of astrophysics. We are particularly interested in low mass galaxies with Mstar < 10^9.25 M_sun. We are particularly interested in two questions:

* **Transient astronomy**: with the rapid growth of high cadence, sensitive, wide-field optical imaging, we now have the ability to study populations of transients in real-time. Dwarf galaxies host a wide range of fascinating, and rare, transients, including superluminous supernovae, gamma-ray bursts, luminous fast blue optical transients (LFBOTs), and potentially Ca-rich transients. Of particular interest, they may host intermediate mass black holes, which can produce flaring through tidal disruption events. The rate of tidal disruption events in dwarf galaxies may depend on stellar population, and could potentially be biased towards dwarfs with less star formation. The luminosities of these transients (rather than their fluxes) also contains substantial information about their origins. Determining the transients of most interest that are worthy of using telescope time to follow-up, as well as measuring the rates of transient populations are fundamental questions in transient astronomy. With a catalog of nearby dwarf galaxies, where we understand the completeness and selection, we can begin to answer some of these questions.
* **Galaxy evolution**: A key constraint on models of galaxy evolution is the quenched fraction of galaxies as a function of stellar mass and environment. Measuring this requires reliable distance or redshift estimates for large samples of low-mass galaxies. However, spectroscopic redshifts are expensive to obtain and are particularly incomplete for quenched dwarf galaxies, which are fainter at fixed stellar mass and therefore harder to target spectroscopically. As a result, current measurements of quenched fractions and environmental trends are limited by selection effects and incompleteness. Galaxy images contain substantial information about distance through surface brightness, angular size, morphology, color gradients, and internal structure. If a model could take a galaxy image cutout as input and predict a redshift, this information could be propagated directly into downstream analyses. These redshift estimates can be folded into measurements of quenched fractions, stellar mass functions, and environmental dependencies. More broadly, applying such a model to wide-field imaging surveys would enable the construction of a three-dimensional map of the nearby Universe using imaging data alone. A model that produces accurate and well-characterized redshift estimates with high completeness and purity would therefore be a powerful tool for studies of low-mass galaxy evolution.

For nearby galaxies (z < 0.03), galaxies are often (but not always) spatially resolved in imaging data. The spatially resolved features contain rich distance information that integrated photometric quantities alone cannot recover:

Angular size: Larger apparent sizes typically indicate closer galaxies
Internal structure: Spiral arms, star-forming regions, and morphological features are resolved
Surface brightness fluctuations: The degree of smoothness versus clumpiness encodes distance-dependent information.

These are not the only properties, and may not be the most important. All aspects of the images should be explored. In addition, the integrated photometric properties, which do contain additional information beyond these spatially resolved features, are extracted directly from the image data. This means that the full image is a strictly richer data representation for redshift inference, especially at low redshift. This challenge focuses on leveraging images to estimate redshifts, rather than relying on handcrafted photometric features.

## The problem
We want to develop a methodology that can produce a complete and pure catalog of galaxies within z < 0.03 with photo-z estimates, where completeness and purity are with respect to the spectroscopic redshift catalog. Our main criteria is that  the model should be at least 95% complete and as pure as possible. While you will evaluate the metrics using all galaxies, we are particularly concerned about dwarf galaxies, and so you will test that the performance for dwarf galaxies (identified by the IS_DWARF flag) is also good. The photo-z estimates should be accurate (as defined by the metrics below). We also want to understand the completeness of the photo-z catalog the model will produce as a function of galaxy color, magnitude, and redshift (see required results section below).

### Training dataset overview
Directory: {datapath}
The dataset is here: galaxy_images_128_v2_train.h5 and a validation set is given here: galaxy_images_128_v2_val.h5.

The dataset contains galaxies from the Dark Energy Spectroscopic Instrument (DESI) Data Release (DR1) with r-band magnitude of r < 21.2 .

The sample is based on the Darragh-Ford z < 0.03 photometric cuts. The resulting training sample is
* A population of dwarf galaxies with Mstar < 10^9.25 M_sun, with stellar mass roughly estimated using prescription from de los Reyes et al. 2025. These will be galaxies that tend to be fainter and at low redshifts.
* Galaxies satisfying Darragh-Ford et al. 2023  “z < 0.03 complete” photometric cuts. These photometric cuts were designed to identify galaxies at z < 0.03 with low purity with many higher redshift galaxies ( z ≲ 0.15 ) included in this sample as well.

Basic quality and cleaning cuts were applied to the source catalogs prior to image construction.
The training sample is not necessarily balanced, and you should explore methods of weighting/balancing the dataset without using changing the dataset manually.

#### Image Cutouts
For each galaxy, an image cutout is provided with the following properties:
* Image shape: 3 × 128 × 128
* Each of the 3 layers corresponds to different photometric filters: g, r, z
* Pixel scale: 0.262 arcsec/pixel
* Centered on the galaxy coordinates. 
* These are coadded images are from Legacy Surveys imaging DR9.
* The images are oriented with North towards the top, however, distance is (and should be in your result) distance invariant.

Note that there will be other background galaxies, stars, image artifacts in the image cutouts, however, as noted before, the cutouts are centered (although the centering algorithm may be imperfect) on the galaxy whose redshift we know.

Each galaxy has a cutout of size 128x128, however, the same performance (and better memory usage) could be achieved with a smaller cutout like 64x64. However, we are not sure which cutout size is better.

The image cutouts are already fully reduced, flux calibrated and science ready, so when training the model you can either use them directly, or convert them to rgb images using the color transformation provided in sdss_rgb function below. Many previous ML methods on galaxy images transform these images to RGB images where the different filters are combined using the color transformation presented in the sdss_rgb function. Use raw g/r/z channels for training by default; sdss_rgb is for visualization unless you justify otherwise.
```
def sdss_rgb(imgs, bands=["g","r","z"], scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)),m = 0.03):
    '''
    Function to convert the 3x128x128 images into RGB images using a color transformation
    '''
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    
    return rgb
```

## Metadata

Each image is associated with catalog-level metadata including:
* DESI object identifier TARGETID (int64)
* Right ascension (RA) in degrees (float64)
* Declination (DEC) in degrees (float64)
* Spectroscopic redshift (Z) (float32)
* Magnitude of galaxy in the g-band (MAG_G) (float32)
* Magnitude of galaxy in the r-band (MAG_R) (float32)
* Magnitude of galaxy in the z-band (MAG_Z) (float32)
* Boolean flag indicated whether the galaxy is a dwarf galaxy or not (IS_DWARF) (boolean)

Do not use the IS_DWARF or Z metadata as an input feature, these are only for metric evaluation and producing the required results. Do not use TARGETID,RA,DEC as input features in the training. You may use MAG_G, MAG_R, MAG_Z as input features.

## Task Metrics:

The goal of this challenge is to predict point estimates of galaxy redshift from imaging data. We do not consider full redshift probability distribution functions (PDFs); each model must output a single photometric redshift estimate per galaxy. We are most interested in identifying galaxies within z < 0.03. Model performance is evaluated by comparing the predicted photometric redshift to the true spectroscopic redshift.
* Prediction bias defined as $\langle \frac{\Delta z}{1 + z_{\rm spec}} \rangle, ;$⁠, i.e. the average value of the prediction error.
* Normalized Median Absolute Deviation ($\sigma_{\rm NMAD}$) defined below. This is a robust measure of the spread of prediction errors. ⁠
$$1.4826 \times \mathrm{Median}\left(\left| \frac{\Delta z}{1 + z_{\text{spec}}} - \text{Median}\left(\frac{\Delta z}{1 + z_{\text{spec}}}\right) \right|\right)$$ 
* Fraction of Outliers ($f_{\rm outlier}$) defined as the fraction of photo-z predictions for which $\left| \frac{\Delta z}{1 + z_{\text{spec}}} \right| > 0.05$⁠, i.e. the fraction of cases where the prediction error is very high. We chose the threshold of 0.05 to easily compare our results with other similar works.
* The completeness of the sample of galaxies with photo z < 0.03 (defined as the fraction of all galaxies with zspec < 0.03 in training dataset that have zphot < 0.03 in the final catalog) should be at least 95%. The purity (defined as the fraction of galaxies with zphot < 0.03 that also have zspec < 0.03 in the training dataset) should be as high as possible while ensuring completeness.

## Required results:

We require that you produce certain key plots and test results

* Make a completeness vs. purity curve (like an ROC plot) for selecting z<0.03 galaxies where completeness and purity are defined above in the task metrics section. Make this plot for all the variations we suggest below (split by magnitude ranges, split by color like g-r < 0.3 and g-r > 0.3,and split by is dwarf or not.)
* Test the performance as a function of magnitude range. Evaluate the metrics in each of these bins separately:
| Bin | Range | Rationale |
|-----|-------|-----------|
| A | r < 19 | How model does on bright galaxies |
| B | 19 < r < 20.5 | How model does on slightly fainter galaxies |
| C | 20.5 < r < 21.2 | Faint end of current DESI targeting |
* Test if there is any variation within the bin. 
* Test the performance as a function of galaxy color g-r = MAG_G - MAG_R. Evaluate the metrics in color bins of size 0.1 separately to search for variations for redder vs bluer galaxies.
* Test the performance for dwarf galaxies versus not dwarfs, identified by the IS_DWARF flag.
* Report Purity at Fixed Completeness and Vice Versa: report two standardized metrics, both for the entire sample and per magnitude bin
**Purity at completeness = 95%**: How clean is the sample when we insist on finding nearly all low-z galaxies?
**Completeness at purity = 90%**: How many low-z galaxies do we recover when we insist on a clean sample?
Perform these tests for both dwarfs and the full sample

## Important points to remember:
Follow standard astronomical practices as much as possible, and use standard astronomical codes to compute any needed values rather than writing your own
We have provided magnitude values, but these may be incorrect for the most highly resolved objects (the most nearby objects, but not always). Do not use this information about bad fluxes in the ML method as a discriminatory tool. However for the more compact sources, the provided photometry and color information is accurate. Do not recompute fluxes or the flux calibration between sources on your own.
The image centering may also be wrong for highly resolved objects, so your algorithm should be robust to centering errors.
Be careful about normalizing images. Normalizing individual images will remove flux and color information, which are very important for photometric redshifts. Best practice is to do channel/filter-wise normalization across your entire dataset, as this will preserve the relative filter differences for each galaxy. Do not normalize each image to have zero mean and unit variance independently. Do not rescale each channel of each image independently. Relative colors between filters encode physical information and per-image normalization will degrade model performance. 
Make plots as much as possible. As one example (but you should extend this as much as possible), do not just quote the prediction bias, but make a plot showing the prediction bias as a function of redshift (including for different subsets of the training set, like described in required results).
Try to make apples-to-apples comparisons to past work on similar problems as much as possible, in particular Darragh-Ford et al. 2023 and the other works listed below
All completeness/purity metrics in this task are computed with respect to the training dataset provided, and do not attempt to infer completeness relative to the underlying photometric population.

Background Material
"Target Selection and Sample Characterization for the DESI LOW-Z Secondary Target Program" This paper introduces the DESI LOW-Z Secondary Target Survey, which combines the wide-area capabilities of the Dark Energy Spectroscopic Instrument (DESI) with an efficient, low-redshift target selection method. Their selection consists of a set of color and surface brightness cuts, combined with modern machine learning methods, to target low-redshift dwarf galaxies (z < 0.03) between 19 < r < 21 with high completeness. We employ a convolutional neural network (CNN) to select high-priority targets.
"Extending the SAGA Survey (xSAGA). I. Satellite Radial Profiles as a Function of Host-galaxy Properties" Using spectroscopic redshift catalogs from the SAGA Survey as a training data set, they optimized a convolutional neural network (CNN) to identify z < 0.03 galaxies from more-distant objects using image cutouts from the DESI Legacy Imaging Surveys.
"Photometric redshifts from SDSS images with an interpretable deep capsule network" This work uses a capsule network to train a neural network that is better suited for identifying morphological features of the input images than traditional CNNs. They use a deep capsule network trained on ugriz images, spectroscopic redshifts, and Galaxy Zoo spiral/elliptical classifications of ∼400 000 Sloan Digital Sky Survey galaxies to do photometric redshift estimation. We achieve a photometric redshift prediction accuracy and a fraction of catastrophic outliers that are comparable to or better than current methods for SDSS main galaxy sample-like data sets ( r < 17.8 and z ≲ 0.4 ) while requiring less data and fewer trainable parameters. Code: https://biprateep.de/encapZulate-1/
“Self Supervised similarity search for large scientific datasets ” This paper uses 42 million galaxy images from the latest data release of the Dark Energy Spectroscopic Instrument (DESI) Legacy Imaging Surveys, to train a self-supervised model to distill low-dimensional representations that are robust to symmetries, uncertainties, and noise in each image. This model is very useful for similarity searches. Code: https://github.com/georgestein/ssl-legacysurvey
“DeepDISC-photoz: Deep Learning-Based Photometric Redshift Estimation for Rubin LSST” This paper trains a photo-z estimator on simulated LSST data from the second Data Challenge (DC2). They produce probabilistic photometric redshifts directly from the simulated, multi-band images. They find that their image-based estimator outperforms feature-based estimators, both for point estimates and probabilistic metrics. They find that the signal-to-noise of the images is the primary limiting factor.
Code: https://github.com/LSSTDESC/rail_deepdisc
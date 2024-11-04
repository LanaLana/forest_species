# Dominant forest species classification through multispectral satellite data

Here is the pipeline to classify dominant forest species via high-resolution and medium-resolution remote sensing images.

We presented the entire experiments and the detailed description of the pipeline in our paper https://ieeexplore.ieee.org/abstract/document/9311828 entitled **"Neural-Based Hierarchical Approach for Detailed Dominant Forest Species Classification by Multispectral Satellite Imagery"**

## Forestry data and satellite observations

<img src="https://github.com/LanaLana/forest_species/blob/master/mandrog.png" width="450">

The markup covers the Mandrogsk forestry in Leningrad  Oblast  of  Russia (around 20.000 hectares). Data was collected by the ground based observations in 2018.

The dataset example is available through the link [Forest species data](https://disk.yandex.ru/d/YtU6RyT5DjZszg).  

The data includes a single satellite image from Sentinel-2 with 13 spectral bands (the spatial resolution is 10 m per pixel) and taxation measurements for four dominant species:

- aspen
- birch
- spruce
- pine

We also provide merged maps for deciduous and conifer classes. In addition to forest masks, we share the IDs for individual forest stands, this information can be used for further pipeline development and results analysis.

## Example notebook

The utils for data preprocessing for deep learning training are presented in the [Forest species data](https://disk.yandex.ru/d/YtU6RyT5DjZszg) file.

Example of notebooks for neural networks training are presented in [Forest species data](https://disk.yandex.ru/d/YtU6RyT5DjZszg)

For the educational purposes, we also share the colab notebook with ML pipeline for the dominant species estimation in the setting where the forestry is splitted in individual stands and a model is trained to predict a single label for each stand [Forest species data](https://disk.yandex.ru/d/YtU6RyT5DjZszg)

## Cite in Bibtex

If you find this information useful for your research, please refer to our work. 

```
@article{illarionova2020neural,
  title={Neural-based hierarchical approach for detailed dominant forest species classification by multispectral satellite imagery},
  author={Illarionova, Svetlana and Trekin, Alexey and Ignatiev, Vladimir and Oseledets, Ivan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={14},
  pages={1810--1820},
  year={2020},
  publisher={IEEE}
}
```

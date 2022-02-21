# LIMESegment
Code to fully reproduce benchmark results (and to extend for your own purposes) from the paper:
<p align="center">Sivill, Flach <a href="">LIMESegment: Meaningful, Realistic</a>, AISTATS 2022.</p>

## Goal
The goal of this package is to provide a modular way of adapting LIME to Time Series data. It provides methods for:
<ul>
  <li> Segmenting a time series </li>
  <li> Perturbing a time series </li>
  <li> Measuring similarity between time series </li>
  <li> Putting these all together to find explanations in the form 'This segment was most import for the overall classification'
  </ul>

## Installation

<ul> 
  <li> Clone repository </li>
  <li> Install dependencies 'pip3 install -r requirements.txt' </li>

## Content
<ul>
  <li> models.py - Contains classification models used to test explanations (KNN, CNN and LSTMFCN) </li>
  <li> explanations.py - Contains explanation modules (LIMESegment, Neves et al., )


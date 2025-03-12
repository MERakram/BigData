# BigData

## Overview

This project works with various datasets to analyze and visualize big data related to world heritage sites, countries, and happiness indexes. It's designed to process, analyze, and derive insights from large volumes of structured and unstructured data, making it a true big data project with focus on data integration, processing, and analytics at scale.

## Installation

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Setup using Poetry

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd bd
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Data Sources

This project uses the following external data sources:

### Countries API

Real-time country data retrieved programmatically through REST API integration.

- Source: [REST Countries API](https://restcountries.com/)
- Features: Country information, flags, languages, currencies, regional blocs, and more
- Integration: Direct API calls with data transformation pipelines

### UNESCO World Heritage Sites

A comprehensive dataset of UNESCO world heritage sites with information about their locations, dates of inscription, and categories.

- Source: [Kaggle - UNESCO World Heritage Sites](https://www.kaggle.com/datasets/ujwalkandi/unesco-world-heritage-sites?select=whc-sites-2019.csv)
- Features: Site names, countries, coordinates, date of inscription, categories, regions

### Countries of the World 2023

Detailed information about countries including demographic, economic, and geographical data.

- Source: [Kaggle - Countries of the World 2023](https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023)
- Features: Population, GDP, area, capitals, languages, currencies, etc.

### World Happiness Report

Dataset containing happiness scores and rankings based on various life factors.

- Source: [Kaggle - World Happiness](https://www.kaggle.com/datasets/unsdsn/world-happiness/data)
- Features: Happiness scores, GDP per capita, social support, health metrics, freedom metrics

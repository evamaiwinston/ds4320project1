# DS 4320 Project 1: California Traffic and Air Quality

This repository contains the data pipeline, analysis, and modeling code for a study examining whether daytime traffic patterns in Los Angeles (Caltrans District 7) are predictive of nighttime NO₂ air quality levels. Using hourly traffic data from Caltrans PeMS and NO₂ measurements from the EPA AQS, a Random Forest Regressor was trained on a lagged daily dataset pairing previous day traffic activity with following night NO₂ concentrations. The repository includes all data processing, exploratory analysis, model tuning, and evaluation code, along with documentation of data provenance, modeling decisions, and results.

Eva Winston  
vxm2ek

[DOI](https://github.com/evamaiwinston/ds4320project1/blob/main/pipeline_main.ipynb)

[Press Release](https://github.com/evamaiwinston/ds4320project1/blob/main/pressrelease.md)

[Data](https://myuva-my.sharepoint.com/:f:/g/personal/vxm2ek_virginia_edu/IgASRpSRGnZiQZVZQA55sCqsAQd5f3gI0EPNJEDve9bN7no?e=fi3VqU)

[Pipeline](https://github.com/evamaiwinston/ds4320project1/blob/main/pipeline_main.ipynb)

[License](LICENSE)

## Problem Definition
Initial problem: Predicting air quality  
Refined problem: In Los Angeles, California, are traffic patterns predictive of the air's NO2 levels? 

Diverging away from the intial problem, I considered the different factors that might contribute to air quality and pollution- factors that may be related to energy generation, agriculture, residential living, transportation, or natural disaster. I also thought about where it might be useful to predict air quality. Big cities tend to have compromised air quality as a result of dense populations. A particular city that is often criticized for its "smoggy air" and its impossible traffic is Los Angeles, California. My thinking began to diverge as I recognized that these two qualities are both heavily associated with LA. The sprawling nature of the city requires a car to get around, unlike other large cities like New York or Chicago that have efficient public transit systems. Thus, I refined my problem down to examining how traffic patterns may be used to predict air quality in LA, specifically NO2 levels because they are a result of vehicle combustion and a key indicator of smog. 

I was drawn to the initial problem about predicting air quality because I feel strongly about treating the environment responsibly. I have worked on projects pertaining to data centers' emissions and environmental impact, but I wanted to look into other factors that may be indicative of air quality. Also, I recently visited LA. I enjoyed my time, so I wanted to a project that could tie in California somehow. While I was there, I heard *a lot* of comments about the incredible volume of traffic in the city and how it takes at least thirty minutes to get anywhere. I learned that in the 60s there was a developing regional tram system in LA, but tire companies lobbied to have it shut down. I was motivated to demonstrate the potential correlation between LA traffic patterns and air quality in hopes of encouraging alternative, more environmental modes of transportation. 

* Headline of press release and (link)

## Domain Exposition 

| Term | Definition |
|------|------------|
| PeMS | Performance Measurement System - Caltrans' platform collecting real-time traffic data from sensors statewide |
Sensor | In-pavement loop detector or radar device that captures real-time traffic information at a fixed roadway location
 Flow | Number of vehicles passing a sensor per unit time |
Occupancy | Percentage of time a sensor is occupied by a vehicle; proxy for congestion
Speed | Average vehicle speed (mph) recorded at a sensor
 EPA AQS | Environmental Protection Agency Air Quality System - the national database storing ambient air pollution measurements 
 Monitor Station | Fixed ground-level sensor that measures ambient pollutant concentrations
NO₂ | Nitrogen Dioxide - a traffic-related pollutant used as a marker for combustion emissions
ppb | Units for NO₂ concentration (parts per billion)
RMSE | Root Mean Squared Error - average prediction error in original units (ppb)
R² | Proportion of variance in NO₂ explained by the model

Los Angeles recorded its first major smog even in 1943, and within the next couple decades scientistis attributed automobiles as the primary cause. This finding has since influenced many environmental policies for the city of LA. However, LA continued to build their freeway system outward, cementing themselves as a car-dependent city thanks to its sprawling nature. Vehicle emissions have remained an air quality concern for residents of LA. To examine this relationship, this dataset draws on two publicly available data sources: Caltran's PeMS and EPA's AQS. PeMS collects real time traffic measurements, such as vehicle counts, speed, and occupany, from loop detecters embedded across the freeway network. The AQS is a national database that includes ambient pollutant concetrations, like NO₂, recorded at fixed ground level monitor stations. Together, these sources allow for an examination of how traffic acitvity on LA's freeways may relate to corresponding nitrogen dioxide levels across the region.

[Link to OneDrive folder of background readings](https://myuva-my.sharepoint.com/:f:/g/personal/vxm2ek_virginia_edu/IgBI2gpYt646Q6daB4rKaXMzAT4__scMmM4LcjL3jeK6YQE?e=3iK9Gy)

| Reading | Description | Link |
|---|---|---|
| Air Pollution In Los Angeles | An overview of LA's ongoing air quality struggles, attributing poor conditions to heavy vehicle traffic, port emissions, heat waves, and wildfires | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQDcTC-u6To4R4pViBmMQ5u3AYhd5rAnRbxk9DDBzFC53mc?e=AuMcfE) |
| PeMS User Guide | Technical documentation providing guidance on how to access, interpret, and work with traffic data collected from Caltrans' statewide network of roadway sensors | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQAC2RleGTuTTa2rMGNNWtfnATwZxYApGBGD9aaPb23trNc?e=6pwdPP) |
| Air pollution and health risks due to vehicle traffic | A modeling study demonstrating that traffic congestion significantly increases roadside NO₂ concentrations and associated health risks, including emergency visits and mortality, for both on-road and near-road populations | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQAIP-7IUFNxRJInk9UlyNBTATCzG5L8KrKAXQI1fhm9xqM?e=44etuD) |
| History - California Air Resources Board | A history of California's air quality regulation, tracing the origins of LA smog back to 1943 and Dr. Haagen-Smit's 1950s discovery that automobile exhaust was the primary cause | [link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQDvFTKs8bGWQ5r_3F5xPGtYAZVs997l4ljbHYREHOazrBU?e=Ex00B0) |
| Analysis of nitrogen oxide emissions from modern vehicles | An experimental analysis examining air quality levels produced by modern vehicle engines running on hydrogen, natural gas, and synthetic fuels| [link](https://myuva-my.sharepoint.com/:u:/g/personal/vxm2ek_virginia_edu/IQDLBXEcsacLQ5a_0jDyAX4sAYaxs57OEpX4DSHQVAIM3CM?e=4mIPh4) |

## Data Creation

The raw data for this dataset comes from two sources. The traffic data is obtained for the Performance Measurement System (PeMS) Data Source from The California Department of Transportation (Caltrans). I applied for an account with PeMS, and then used the Data Clearinghouse tabe and filtered the search to hourly station data in LA's district, District 7. I manually downloaded gzippedfiles to my computer from each month of 2024 and 2025. To get the metadata on sensor locations, I filtered the search to station metadata, and downloaded one gzipped file since the stations are consistent from year to year. I acquired the air quality data from the United States Environmental Protection Agency. On their AirData page, I navigated to the Pre-Generated Data Files tab. Here, I selected 'Tables of Hourly Data' and downloaded the NO2 zip files for 2024 and 2025. I also downloaded the Monitor Listing for spatial and temporal metadata about the monitors used to measure NO2 levels. I moved all of these files into the data/raw folder of my project directory. 

| File | Description | Code |
|---|---|---|
| `stations.parquet` | Traffic station metadata from PeMS  | [Code](https://github.com/evamaiwinston/ds4320project1/blob/main/pipeline_main.ipynb#2.-Stations-Metadata-to-stations.parquet) |
| `traffic.parquet` | PeMS hourly traffic sensor readings | [Code](https://github.com/evamaiwinston/ds4320project1/blob/main/pipeline_main.ipynb#3.-PeMS-Hourly-Traffic-to-traffic.parquet) |
| `air_quality.parquet` |  EPA AQS hourly NO₂ concentration measurements | [Code](https://github.com/evamaiwinston/ds4320project1/blob/main/pipeline_main.ipynb#4.-EPA-AQS-Hourly-NO₂-to-air_quality.parquet) |
| `monitors.parquet` | EPA AQS air quality monitor station metadata | [Code](https://github.com/evamaiwinston/ds4320project1/blob/main/pipeline_main.ipynb#5.-EPA-AQS-Monitors-Metadata-to-monitors.parquet) |
 
Several sources of bias may have been introduce during data collection. Only data for 2024 and 2025 was collected, which may not be representative of longer term traffic and air quality patterns. Factors like unusual weather years, traffic patterns caused by the pandemic, or infrastructure changes aren't accounted for and could limit generizability. Nitrogen dioxide was selected as the sole air quality target. However, traffic emissions can contribute to other pollutants including CO, particulate matter, or ozone that all collectively contribute to smog. Focusing on nitrogen dioxide alone may understate the full impact of traffic on air quality. Also, spatial linkage between traffic stations and air quality monitors assumes that nearby stations are measuring the same air mass, which may not always be true. Monitors located downwind, upwind, different elevations, etc. may distort the relationship. The predetermined locations of monitors and traffic stations across CA's District 7 may not all be evenly distributed, which would introduce bias against certain corridors and communities that may be underrepresented in the analysis. 

Several steps were taken to mitigate potential sources of bias. During analysis, station-monitor pairs were restricted to only pairs within 1km of each other. This was done in hopes of reducing the spatial bias introduced by assuming distant sensors are measuring the same air mass, and ensuring traffic readings are more likely to reflect the conditions at the monitoring site. In terms of temporal scope, expanding the dataset to cover a longer historical period could improve generalizability. However, focusing on 2024-2025 may offer its own value in focusing on recent conditions. Future work could expand the pollutant scope beyond nitrogen dioxide to include other pollutants to provide a more compreshensive assessment of how traffic contributes to overall air quality and smog in Los Angeles. 

Critical decisions were made throughout the analysis that warrant justification. NO₂ was selected as the target pollutant because research has demonstrated it is a direct byproduct of internal combustion engines. It is widely used in the literature as a primary indicator of trafficrelated air pollution, unlike particulate matter or ozone which have many formation pathways. The choice to filter station-monitor pairs to within 1km was a deliberate tradeoff between sample size and spatial validity. While this reduced the dataset significantly, pairing a traffic station with a monitor 10-20km away, or more, introduces substantial uncertainty when assuming local traffic emissions are linked to NO₂ levels. Sources like wind or topography could easily confound this relationship. A main decision during analysis was to structure the problem as a lagged daily prediction task. This decision was made because during daylight hours, solar radiation drives photochemical reactions that continuously break down NO₂ and limits its accumulation. After sunset, this photochemical dispersion ceases and the planetary boundary layer collapses, trapping pollutants near the surface. The lagged structure aims to determine if NO₂ emitted by daytime traffic accumulates overnight rather than dispersing, linking the previous day's traffic activity directly to nighttime NO₂ concentrations. 

The choice of Random Forest over linear regression was justified both by the expectation of non-linear interactions between traffic volume, speed, occupancy, and NO₂, and by the weak linear correlations observed during exploratory analysis. Finally, to adress a common source of uncertainty in modeling, GridSearchCV with 5-fold cross-validation was used as well as a strictly held out test set for evaluation. 


## Metadata
[View ER at the logical level here](https://myuva-my.sharepoint.com/:i:/g/personal/vxm2ek_virginia_edu/IQDIqTL6H61ARZR1v4FWQ4vwAc7WoSs87bTqtWMnbqbH6jk?e=gDdY8N)

| Table | Description | Link |
|-------|-------------|------|
| stations | Caltrans traffic sensor station metadata including location, freeway, and lane info | [stations.csv](https://myuva-my.sharepoint.com/:x:/g/personal/vxm2ek_virginia_edu/IQBL85vqCrzvTbEgG_0xnvt3AXC8AJyFKnGmrMjS6_LRz24?e=xJRu6K) |
| traffic | Hourly traffic measurements per station including flow, speed, and occupancy | [traffic.csv](https://myuva-my.sharepoint.com/:f:/g/personal/vxm2ek_virginia_edu/IgASRpSRGnZiQZVZQA55sCqsAQd5f3gI0EPNJEDve9bN7no?e=51w3U5) |
| monitors | NO₂ air quality monitor metadata including location and site info | [monitors.csv](https://myuva-my.sharepoint.com/:x:/g/personal/vxm2ek_virginia_edu/IQCixXADlje7RJVLX0lXmmvDAYM-yQYP4thu6YyBB12yKOw?e=lNZPDQ) |
| air_quality | Hourly NO₂ readings per monitor | [air_quality.csv](https://myuva-my.sharepoint.com/:x:/g/personal/vxm2ek_virginia_edu/IQCBQsD8tWpKR4E7UqrqVGWbARkwWAhKd46XHIKIGNQa5E0?e=9ZeB6F) |
| station_monitor_mapping | Spatial join table linking traffic stations to nearby air quality monitors | [station_monitor_mapping.csv](https://myuva-my.sharepoint.com/:x:/g/personal/vxm2ek_virginia_edu/IQCsVjYP38eZQapcxPOLJ3aAAdZXtv05rhq6THPlb6pg_Ys?e=ZmjfyK) |
| analysis | Joined table combining traffic and air quality data by station-monitor pairs | [analysis.csv](https://myuva-my.sharepoint.com/:f:/g/personal/vxm2ek_virginia_edu/IgASRpSRGnZiQZVZQA55sCqsAQd5f3gI0EPNJEDve9bN7no?e=51w3U5) |
| analysis_features | Final modelling table with added temporal features (hour, day, month, year) | [analysis_features.csv](https://myuva-my.sharepoint.com/:f:/g/personal/vxm2ek_virginia_edu/IgASRpSRGnZiQZVZQA55sCqsAQd5f3gI0EPNJEDve9bN7no?e=51w3U5) |


| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMP | Date and hour of the observation |
| station_id | BIGINT | Unique identifier for the Caltrans traffic sensor station |
| total_flow | DOUBLE | Total number of vehicles recorded at the station during the hour |
| avg_speed | DOUBLE | Average speed of vehicles at the station (mph) |
| avg_occupancy | DOUBLE | Average lane occupancy — proportion of time a vehicle is detected over the sensor |
| station_lat | DOUBLE | Latitude of the traffic sensor station |
| station_lon | DOUBLE | Longitude of the traffic sensor station |
| monitor_lat | DOUBLE | Latitude of the paired NO₂ air quality monitor |
| monitor_lon | DOUBLE | Longitude of the paired NO₂ air quality monitor |
| distance_km | DOUBLE | Distance in kilometers between the traffic station and paired air quality monitor |
| county | VARCHAR | County name where the station-monitor pair is located |
| city | VARCHAR | City name where the station-monitor pair is located |
| no2_ppb | DOUBLE | NO₂ concentration measured at the paired air quality monitor (parts per billion) |
| hour | BIGINT | Hour of day extracted from timestamp (0–23) |
| day_of_week | BIGINT | Day of week extracted from timestamp (0=Monday, 6=Sunday) |
| month | BIGINT | Month extracted from timestamp (1–12) |
| year | BIGINT | Year extracted from timestamp |

| Column | Min | Max | Mean | Std Dev | Notes |
|--------|-----|-----|------|---------|-------|
| total_flow | 0.0 | 17381.0 | 3612.55 | ~2800 | Vehicles per hour |
| avg_speed | 3.0 | 85.0 | 60.57 | 11.03 | mph; low values indicate congestion |
| avg_occupancy | 0.0 | 0.97 | 0.09 | 0.07 | Proportion of time a vehicle is detected over the sensor |
| distance_km | 0.16 | 43.72 | 7.38 | 4.13 | Distance between station and monitor; filtered to ≤1km for modelling |
| no2_ppb | 0.0 | 53.1 | 12.4 | 9.79 | Parts per billion; zero values excluded from final model |
| day_of_week | 0 | 6 | — | — | 0=Monday, 6=Sunday |
| month | 1 | 12 | — | — | 1=January, 12=December |
| year | 2024 | 2025 | — | — | Two-year observation window |






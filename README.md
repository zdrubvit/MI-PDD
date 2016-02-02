A semestral project for MI-PDD.

This script predicts five most probable first booking destinations for new users on Airbnb website, based on the data collected from their currently registered users.

Folder "data" contains the two datasets with training and testing users. All the other datasets can be found at
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data
Only the users are used by the script. Sessions dataset is quite big to appear here.

submission.csv contains the final submission data - every line represents one user (defined by his id) and his probable destitation for booking. The bookings are in a descending order, from the most probable. There are only a few users, the whole file is too big.

# Model inception to deployment example

[Source](https://medium.com/datadriveninvestor/from-model-inception-to-deployment-adce1f5ed9d6)

Run ```docker-compose build``` and then ```docker-compose up``` to run

## Test 1 

Then go to website http://192.168.99.100:8000 per nginx.conf file (the docker-machine virtual ip)

## Test 2 api
 
Use Postman 
http://192.168.99.100:8000/predict


```bash
curl --location --request POST 'http://192.168.99.100:8000/predict' \
--header 'Content-Type: application/json' \
--data-raw '[
    {
        "sepal_length": 6.3,
        "sepal_width": 2.3,
        "petal_length": 4.4,
        "petal_width": 1.3
    } , 
    {
        "sepal_length": 6.3,
        "sepal_width": 2.3,
        "petal_length": 4.4,
        "petal_width": 5.3
    }
]'
```
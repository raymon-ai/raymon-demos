# Running Raymon examples

## Local machine: Docker

1. Run `setup_project.py` with `DEMO` set to the demo you want to run. This will create a project (if none with the given name exists yet), will set the project manifest and will set a model profile.

2. Copy the project id that is printed to stdout and set the `PROJECT_ID` environment variable in `docker-compose.yml` and / or `job.yaml` file.

3. Run `docker-compose build && docker-compose up` in the relevant folder. You might be asked to login!

## Running on Kubernetes

1. For k8s, you need to create the secret first:
    a. Prod:   
`kubectl create secret generic retinopathy-secret --from-file=m2mcreds-retinopathy.json`
or
`kubectl create secret generic houseprices-secret --from-file=m2mcreds-houseprices.json`
b. Staging:
`kubectl create secret generic retinopathy-secret-staging --from-file=m2mcreds-retinopathy.staging.json`
or
`kubectl create secret generic houseprices-secret-staging --from-file=m2mcreds-houseprices.staging.json`

2. `kubectl apply -f job.yaml`



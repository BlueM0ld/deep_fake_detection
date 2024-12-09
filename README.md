This project is a work in progress - as it has only run on 11GB of data.

Can be accessed her http://65.21.107.207:8501/

Please note train5.py is the most up to date training file need to clean up!

Create docker images - need to navigate to the folders

- fastapi
    `DOCKER_BUILDKIT=0 docker buildx build . -t fastapi `

- streamlit 
    `DOCKER_BUILDKIT=0 docker buildx build . -t streamlit `
- postgres
    `DOCKER_BUILDKIT=0 docker buildx build . -t postgres `


Then from root execute docker compose
    `docker-compose up `


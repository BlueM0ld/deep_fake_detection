#base image
FROM postgres:latest

# environment variables
ENV POSTGRES_PASSWORD=docker
ENV POSTGRES_DB=db

# copy initialization script to the entrypoint directory
COPY db.sql /docker-entrypoint-initdb.d/

#expose port
EXPOSE 5432

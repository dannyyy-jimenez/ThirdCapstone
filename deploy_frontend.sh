echo "Deploying frontend..."
cd frontend
export REACT_APP_API_URL=/api
npm run build
aws s3 sync build/ s3://phonetix-frontend --acl public-read

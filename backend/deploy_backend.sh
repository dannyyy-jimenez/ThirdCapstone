echo "Deploying backend..."
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q0n7e1b0
docker build -t phonetix-prod-backend .
docker tag phonetix-prod-backend:latest public.ecr.aws/q0n7e1b0/phonetix-prod-backend:latest
docker push public.ecr.aws/q0n7e1b0/phonetix-prod-backend:latest
cd aws_deploy
eb deploy

#!/bin/bash

# AWS EC2 Development Environment Setup for RL Codebase (Ubuntu)
# This script creates and configures an Ubuntu EC2 instance for remote development

set -e

# Configuration
INSTANCE_TYPE="g4dn.xlarge"  # GPU instance for ML workloads
AMI_ID="ami-0916509d3ddc82e36" # Ubuntu AMI
KEY_NAME="dev-key"
SECURITY_GROUP_NAME="dev-sg"
INSTANCE_NAME="rl-dev-server"

echo "Setting up AWS development environment with Ubuntu..."

# Create key pair if it doesn't exist
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME >/dev/null 2>&1; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "Key saved to ~/.ssh/${KEY_NAME}.pem"
else
    echo "Key pair $KEY_NAME already exists"
fi

# Get default VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
echo "Using VPC: $VPC_ID"

# # Create security group if it doesn't exist
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME >/dev/null 2>&1; then
    echo "Creating security group: $SECURITY_GROUP_NAME"
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for development server" \
        --vpc-id $VPC_ID \
        --query 'GroupId' --output text)

    # Add SSH access
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0

    # Add VSCode server port
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 8080 \
        --cidr 0.0.0.0/0
else
    echo "Security group $SECURITY_GROUP_NAME already exists"
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --query 'SecurityGroups[0].GroupId' --output text)
fi

echo "Using security group: $SECURITY_GROUP_ID"

# Check if instance already exists
EXISTING_INSTANCE=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,stopped,stopping,pending" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")

if [ "$EXISTING_INSTANCE" != "None" ] && [ "$EXISTING_INSTANCE" != "" ]; then
    echo "Found existing instance: $EXISTING_INSTANCE"
    INSTANCE_ID=$EXISTING_INSTANCE

    # Get current state
    INSTANCE_STATE=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].State.Name' --output text)

    echo "Instance state: $INSTANCE_STATE"

    if [ "$INSTANCE_STATE" = "stopped" ]; then
        echo "Starting existing instance..."
        aws ec2 start-instances --instance-ids $INSTANCE_ID
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    elif [ "$INSTANCE_STATE" = "stopping" ]; then
        echo "Waiting for instance to stop, then starting..."
        aws ec2 wait instance-stopped --instance-ids $INSTANCE_ID
        aws ec2 start-instances --instance-ids $INSTANCE_ID
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    elif [ "$INSTANCE_STATE" = "pending" ]; then
        echo "Waiting for pending instance to be running..."
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    else
        echo "Instance is already running"
    fi
else
    # Create new EC2 instance
    echo "Creating new EC2 instance..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SECURITY_GROUP_ID \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --query 'Instances[0].InstanceId' --output text)

    echo "Instance created: $INSTANCE_ID"
    echo "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
fi

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance is ready!"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "SSH command: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Instance management commands:"
echo "  Stop instance:      aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "  Start instance:     aws ec2 start-instances --instance-ids $INSTANCE_ID"
echo "  Terminate instance: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""

# # Create setup script for the Ubuntu instance
# cat > setup-ubuntu-instance.sh << 'EOF'
# #!/bin/bash
# # This script runs on the Ubuntu EC2 instance to set up the development environment

# set -e

# echo "Updating system packages..."
# sudo apt update && sudo apt upgrade -y

# echo "Installing development tools..."
# sudo apt install -y build-essential git curl wget htop vim unzip software-properties-common

# echo "Installing Python 3 and pip..."
# sudo apt install -y python3 python3-pip python3-venv python3-dev

# echo "Installing Node.js (for VSCode server)..."
# curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
# sudo apt install -y nodejs

# echo "Installing NVIDIA drivers and CUDA (for GPU support)..."
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# sudo dpkg -i cuda-keyring_1.0-1_all.deb
# sudo apt update
# sudo apt install -y cuda-drivers cuda-toolkit-12-8
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# echo "Installing Docker (optional, for containerized development)..."
# curl -fsSL https://get.docker.com -o get-docker.sh
# sudo sh get-docker.sh
# sudo usermod -aG docker ubuntu

# echo "Installing VSCode server..."
# curl -fsSL https://code-server.dev/install.sh | sh
# sudo systemctl enable --now code-server@ubuntu

# echo "Configuring code-server..."
# mkdir -p ~/.config/code-server
# cat > ~/.config/code-server/config.yaml << 'CODESERVEREOF'
# bind-addr: 0.0.0.0:8080
# auth: password
# password: change-this-secure-password
# cert: false
# CODESERVEREOF

# echo "Installing MongoDB (for fMRI data)..."
# wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
# echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# sudo apt update
# sudo apt install -y mongodb-org
# sudo systemctl enable --now mongod

# echo "Creating workspace directory..."
# mkdir -p ~/workspace
# cd ~/workspace

# echo "Setting up Python virtual environment..."
# python3 -m venv rl_env
# source rl_env/bin/activate
# echo "source ~/workspace/rl_env/bin/activate" >> ~/.bashrc

# echo "Installing common ML/RL packages..."
# pip install --upgrade pip
# pip install numpy scipy matplotlib jupyter torch torchvision tensorflow gym opencv-python

# echo "Restarting code-server..."
# sudo systemctl restart code-server@ubuntu

# echo "Setup complete!"
# echo "Access VSCode at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
# echo "Default password: change-this-secure-password (CHANGE THIS!)"
# echo "Python virtual environment: ~/workspace/rl_env"
# echo "To activate: source ~/workspace/rl_env/bin/activate"
# EOF

# echo ""
# echo "Next steps:"
# echo "1. First configure AWS CLI: aws configure"
# echo "2. Run this script: ./setup-aws-dev.sh"
# echo "3. Connect to your instance: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@[PUBLIC_IP]"
# echo "4. Run the setup script on the instance: bash setup-ubuntu-instance.sh"
# echo "5. Clone your repository and install project requirements"
# echo "6. Access VSCode at: http://[PUBLIC_IP]:8080"
# echo ""
# echo "Ubuntu setup script saved as: setup-ubuntu-instance.sh"

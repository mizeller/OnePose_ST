echo "Enabling non-root docker commands"
echo "Should automatically docker run hello-world"

chown root:docker /var/run/docker.sock
chown -R root:docker /var/run/docker

docker run hello-world

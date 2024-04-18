rsync --progress --exclude 'venv' -aP ./ s_01hvrd5jfkwt0znmkf9njexra7@ssh.lightning.ai:/teamspace/studios/this_studio/
while inotifywait -r -e modify,create,delete ./; do
    rsync --progress --exclude 'venv' -aP ./ s_01hvrd5jfkwt0znmkf9njexra7@ssh.lightning.ai:/teamspace/studios/this_studio/ 
done

destination=s_01hvrd5jfkwt0znmkf9njexra7@ssh.lightning.ai
folder=/teamspace/studios/this_studio/
rsync --progress --exclude 'venv' -aP ./ $destination:$folder
while inotifywait -r -e modify,create,delete ./; do
    rsync --progress --exclude 'venv' -aP ./ $destination:$folder
done

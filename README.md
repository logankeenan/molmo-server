# molmo server

Leverages [allenai/Molmo-7B-O-0924](https://huggingface.co/allenai/Molmo-7B-O-0924)

### prompts
* "center point coordinate of the <object>. json output format only x,y"
* "find the sign up button. point coordinates only"


### Running locally

```bash
gunicorn --bind 0.0.0.0:5000 main:app
```

Run show ssh and keep running
```bash
nohup gunicorn --bind 0.0.0.0:5000 main:app &
```

get the pid in order to kill the process
```bash
 ps aux | grep 'gunicorn --bind 0.0.0.0:5000 main:app'
```

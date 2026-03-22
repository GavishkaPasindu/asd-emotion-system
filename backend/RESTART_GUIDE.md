# ⚠️ Very Important (After you restart):

When you start the instance again, AWS will give you a **NEW Public IP address**.

Because the IP changes, you must do these 3 things:

1. **Update Duck DNS**: Copy the NEW Public IP from AWS and paste it into Duck DNS (update IP).
2. **Update Vercel**: If the IP changed, Vercel might need a minute to reconnect (or you might need to check the Duck DNS link again).
3. **Restart Backend**: You will need to SSH into the instance again and run the Gunicorn Timeout command:

```bash
cd asd-emotion-system/backend
source venv/bin/activate
gunicorn --worker-class eventlet -w 1 --timeout 600 -b 0.0.0.0:5000 -D app:app
```

**Wait for it to say "Stopped" first, then Start it! 🛑➡️🟢**

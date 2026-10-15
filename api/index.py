"""
Vercel Serverless Function Handler
==================================
WSGI bridge routing incoming HTTP requests to the Flask application instance.
"""

from app import app

# WSGI handler for Vercel Serverless Python Runtime
app_handler = app

if __name__ == "__main__":
    app.run()

import functions_framework
from sd import sd

# Register an HTTP function with the Functions Framework
@functions_framework.http
def  http_function(request):
  
  # Your code here
  images = sd.predict_sd(
    project="554514247611",
    endpoint_id="5563924660732559360",
    location="us-west1",
    instances= request.get_json(silent=True)
)
  # Return an HTTP response
  return images[0]

from django.db import models

# Create your models here.
class usermodel(models.Model):
    username=models.CharField(primary_key=True,max_length=20)
    useremail=models.CharField(max_length=50)
    password=models.CharField(max_length=10)
   
    def __str__(self):
        return self.username
   

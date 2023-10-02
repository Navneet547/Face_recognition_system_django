from rest_framework import serializers
from app1.models import usermodel

class userloginSerializer(serializers.ModelSerializer):
    username=serializers.CharField(max_length=20)
    # useremail=models.CharField(max_length=50)
    password=serializers.CharField(max_length=10)

    class Meta:
      model=usermodel
      fields=['username','password']


class usersignupSerializer(serializers.ModelSerializer):
    username=serializers.CharField(max_length=20)
    useremail=serializers.CharField(max_length=50)
    password=serializers.CharField(max_length=10)

    class Meta:
      model=usermodel
      fields=['username','useremail','password']

class userforgotSerializer(serializers.ModelSerializer):
    useremail=serializers.CharField(max_length=50)
    class Meta:
      model=usermodel
      fields=['useremail']

class userotpSerializer(serializers.ModelSerializer):
   otp=serializers.CharField(max_length=4)
   class Meta:
        model=usermodel
        fields=['otp']
        
class userresetSerializer(serializers.ModelSerializer):
    new_password = serializers.CharField(max_length=128)
    confirm_new_password = serializers.CharField(max_length=128)
    class Meta:
        model=usermodel
        fields=['new_password','confirm_new_password']
        
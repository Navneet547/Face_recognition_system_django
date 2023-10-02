# views.py
import cv2
import numpy as np
from PIL import Image
import os
from django.http import HttpResponse
from django.views import View

from django.shortcuts import render
from django.conf import settings


class AddFacesView(View):
    def get(self, request):
        return render(request, 'add_face.html')

    def post(self, request):
        id = request.POST.get('idInput')
        name = request.POST.get('nameInput')
    
        path = "/home/admin1/Downloads/akash.mp4"
        video = cv2.VideoCapture(path)
        facedetect = cv2.CascadeClassifier('/home/admin1/Desktop/djangofacedetection/project1/app1/static/images/haarcascade_frontalface_default.xml')
        count = 0
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                file_path=f"/home/admin1/Desktop/djangofacedetection/project1/app1/static/data/user.{id}.{count}.jpg"
                cv2.imwrite(file_path, gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if count >500:
                break
        video.release()
        cv2.destroyAllWindows()
       
        self.update_mapping_file(id, name)  # Update the mapping file with the ID, name, and file path

        self.train_model()  # Trigger the training of the face recognition model after adding new faces
        # return HttpResponse(status=200)
        return redirect('add_faces')
        # return render(request,'detect_faces.html',{"message":"Dataset Collection Done.................."})

    def update_mapping_file(self, id, name):
        mapping_file = "/home/admin1/Desktop/djangofacedetection/project1/app1/static/training/mapping.txt"
        with open(mapping_file, "a") as file:
            file.write(f"{id} {name}\n")
            # {file_path}

    def train_model(self):
   

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static','data')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        def getImageID(path):
            imagePath = [os.path.join(path, f) for f in os.listdir(path)]
            faces=[]
            ids=[]
            for imagePaths in imagePath:
                faceImage = Image.open(imagePaths).convert('L')
                faceNP = np.array(faceImage)
                if faceNP.shape[0] > 0 and faceNP.shape[1] > 0:
                    Id= (os.path.split(imagePaths)[-1].split(".")[1])
                    Id=int(Id)
                    faces.append(faceNP)
                    ids.append(Id)
                    cv2.imshow("Training",faceNP)
                    cv2.waitKey(1)
            return ids, faces

        IDs, facedata = getImageID(path)
        recognizer.train(facedata, np.array(IDs))
        recognizer.write("Trainer.yml")
        cv2.destroyAllWindows()
        print("Training Completed............")
# views.py

class FaceRecognitionView(View):
    def get(self, request):
        return render(request, 'detect_face.html')

    def post(self, request):
        video = cv2.VideoCapture("/home/admin1/Downloads/group.mp4")
        facedetect = cv2.CascadeClassifier('/home/admin1/Desktop/djangofacedetection/project1/app1/static/images/haarcascade_frontalface_default.xml')

        recognizer_file = "/home/admin1/Desktop/djangofacedetection/project1/Trainer.yml"
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        name_mapping = self.load_mapping_file("/home/admin1/Desktop/djangofacedetection/project1/app1/static/training/mapping.txt")
        
        if os.path.exists(recognizer_file):
            recognizer.read(recognizer_file)  # Load the pre-trained face recognition model
        
        while True:
            ret, frame = video.read()
            
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf <70:
                    name = name_mapping.get(serial, "Unknown")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        video.release()
        cv2.destroyAllWindows()
        return redirect('detect_face')
        # return HttpResponse("Face recogntion.... completed")
    
    def load_mapping_file(self, filename):
        name_mapping = {}
        if os.path.exists(filename):
            with open(filename, "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        id, name = line.split()
                        name_mapping[int(id)] = name
        return name_mapping




from django.shortcuts import render,redirect
from django.views import *
from app1.models import *
from django.core.mail import send_mail
from django.contrib import messages
from django.contrib.auth.models import User
import random

from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.contrib.auth import logout
# from django.contrib.auth.decorators import login_required
# from django.contrib.auth.decorators import login_required


class login_view(View):
    def get(self,request):
        return render(request,"login.html")
    
    def post(self, request):
     
     input_username2 = request.POST.get('username')
     input_password2 = request.POST.get('password')
    
     try:
        user = usermodel.objects.get(username=input_username2)
        
        if input_password2 == user.password:
            messages.success(request,'login successful')
            return redirect('home')
            # return render(request, 'home.html',{'message':messages})
        else:
            return render(request, 'login.html', {"alertpassword": "**wrong password**"})
            
     except usermodel.DoesNotExist:
          return render(request, 'login.html', {"alertemail": "**wrong email id**"})

   
 

class signup_view(View):
    def get(self,request):

        return render(request,"signup.html")

    def post(self,request):
        input_username=request.POST.get('username')
        input_email=request.POST.get('email')
        input_password=request.POST.get('password')
        input_cpassword=request.POST.get('cpassword')
        alert="**missmatch password**"

        if(input_cpassword==input_password):
            usermodel(username=input_username,useremail=input_email,password=input_password).save()
            messages.success(request,'User created successful')
            return  redirect('login',{'message':messages})
            # return  render(request,'login.html',{'message':messages})
        else:
            return render(request,'signup.html',{'alert':alert})


class forgot_view(View):
    def get(self,request):

        return render(request,"forgot.html")
    def post(self,request):
        f_email=request.POST.get('email')
        otp=random.randint(1000,9999)
        
        
        try:
            for_email= usermodel.objects.get(useremail=f_email)
            
            send_mail(
            "OTP Verification",
            f'here is your otp:{otp}',
            "navneet.kumar@indicchain.com",
            [f'{f_email}'],
            fail_silently=False,
            )
            # return render(request, 'otp.html')
            request.session['otp'] = otp 
            request.session['email']=f_email            
            return redirect('otp')
       
        except usermodel.DoesNotExist:
            return render(request, 'forgot.html', {"alertemail": "**wrong email id**"})
           

class otp_view(forgot_view) :
    def get(self,request):        
        return render(request,"otp.html")
    def post(self,request):
        input_otp=request.POST.get('otp')
        otp = request.session.get('otp') 
      
        print(input_otp)
        if(input_otp==str(otp)):            
            return redirect('reset')
            # return render(request,'reset.html')
        
        return render(request,'otp.html',{'otp_alert':'**wrong otp**'})





class reset_view(View):
    def get(self, request):
        
        return render(request, "reset.html")
    
    def post(self, request):
        newpass = request.POST.get('pass1')
        Cnewpass = request.POST.get('Cpass1')        
        email=request.session['email']
        if newpass == Cnewpass:                        
                user = usermodel.objects.get(useremail=email)
                id = usermodel.objects.get(useremail=email).id
                name = usermodel.objects.get(useremail=email).username
               
                usermodel(id=id,username=name,useremail=email,password=newpass).save()
                messages.success(request, "Password updated.")
                return redirect('login')
                # return render(request, 'login.html')        
        else:
            return render(request, 'reset.html', {"notMatch_alert": "Passwords do not match"})



########################################....APIView.....##############################################


from rest_framework.generics import GenericAPIView
from rest_framework.response import Response  
from rest_framework import status  
from drf_yasg.utils import swagger_auto_schema
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate,login
from .serializers import userloginSerializer,usersignupSerializer,userforgotSerializer,userotpSerializer,userresetSerializer


   
class login_APIView(GenericAPIView):
    @swagger_auto_schema(
        operation_description="login users",
        request_body=userloginSerializer
    )
    def post(self, request):
        serializer = userloginSerializer(data=request.data)
        
        if serializer.is_valid():
            username1 = serializer.validated_data['username']
            password1 = serializer.validated_data['password']

        try:
            user = usermodel.objects.get(username=username1)
        
            if password1 == user.password:
                return Response({'message': 'Login successful'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
        except:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class signup_APIView(GenericAPIView):
    @swagger_auto_schema(
        request_body=usersignupSerializer,
         operation_description="Create a new user"
     )
    def post(self, request):
        serializer = usersignupSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"status": "success", "data": serializer.data}, status=status.HTTP_201_CREATED)
        else:
            return Response({"status": "error", "data": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
         
class forgot_APIView(GenericAPIView):
    @swagger_auto_schema(
        request_body=userforgotSerializer,
         operation_description="forgot password"
     )
    def post(self, request):
        otp=random.randint(1000,9999)
        serializer = userforgotSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['useremail']
            request.session['useremail']=email

            try:
                user = usermodel.objects.get(useremail=email)
                
                request.session['otp'] = otp 
                subject = 'Password Reset Request'
                message = f'{ otp }'
                from_email = 'navneet.kumar@indicchain.com'
                recipient_list = [email]
                send_mail(subject, message, from_email, recipient_list)

                return Response({'message': 'Otp sent on email successfully'}, status=status.HTTP_200_OK)
            
            except usermodel.DoesNotExist:
                return Response({'error': 'User with this email does not exist'}, status=status.HTTP_404_NOT_FOUND)

        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)   
   
class OTP_APIView(GenericAPIView):
    @swagger_auto_schema(
        request_body=userotpSerializer,
         operation_description="otp submit here"
     )

    def post(self, request):
        serializer = userotpSerializer(data=request.data)
        
        if serializer.is_valid():
            submitted_otp = serializer.validated_data['otp']
            expected_otp = request.session.get('otp')

            if submitted_otp == str(expected_otp):
                return Response({'message': 'OTP matches successfully'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'OTP not matched'}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class Reset_APIView(GenericAPIView):
    @swagger_auto_schema(
        request_body=userresetSerializer,
         operation_description="reset password here"
     )
  
    def post(self, request):
        serializer = userresetSerializer(data=request.data)
        
        if serializer.is_valid():
            new_password = serializer.validated_data['new_password']
            confirm_new_password = serializer.validated_data['confirm_new_password']
            email=request.session.get('useremail')

            if new_password == confirm_new_password:
                               
                user = usermodel.objects.get(useremail=email)
                # id = usermodel.objects.get(useremail=email).id
                name = usermodel.objects.get(useremail=email).username
               
                usermodel(username=name,useremail=email,password=new_password).save()
                return Response({'message': 'Password reset successful'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Passwords do not match'}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

########################################....End of APIView.....##############################################




class home_view(View):
    def get(self, request):
       
        return render(request, 'home.html')    

# from django.contrib.auth.decorators import login_required
# from django.utils.decorators import method_decorator
# from django.views import View
# from django.shortcuts import render

# @method_decorator(login_required(login_url=''), name='dispatch')
# class home_view(View):
#     def get(self, request):
#         return render(request, 'home.html')


class LogoutView(View):
    def get(self, request):
        logout(request)
        return redirect('login')
        

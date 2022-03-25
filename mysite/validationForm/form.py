from django import forms
from django.forms import ModelForm
from django import forms
from validationForm import *
from validationForm.models import Post

class PostForm(ModelForm):
    class Meta:
        model = Post
        
        fields = ["username", "gender", "text"]
        
    def clean(self):
        super(PostForm, self).clean()
         
        # extract the username and text field from the data
        username = self.cleaned_data.get('username')
        text = self.cleaned_data.get('text')
 
        # conditions to be met for the username length
        if len(username) < 5:
            self._errors['username'] = self.error_class([
                'Minimum 5 characters required'])
        if len(text) <10:
            self._errors['text'] = self.error_class([
                'Post Should Contain a minimum of 10 characters'])
 
        # return any errors if found
        return self.cleaned_data
    
class GeekForms(forms.Form):
    field = forms.CharField(
        error_messages={
            'required':"Please enter nane"
        }
    )
        
        
        
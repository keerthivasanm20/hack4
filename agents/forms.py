from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column
from .models import Agent

class AgentForm(forms.ModelForm):
    class Meta:
        model = Agent
        fields = ['name', 'description', 'skills', 'image']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'skills': forms.TextInput(attrs={'placeholder': 'Enter skills separated by commas'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Row(
                Column('name', css_class='form-group col-md-6 mb-3'),
                Column('image', css_class='form-group col-md-6 mb-3'),
            ),
            'description',
            'skills',
            Submit('submit', 'Create Agent', css_class='btn btn-primary')
        )

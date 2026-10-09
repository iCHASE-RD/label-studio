"""Regression coverage for annotation writes interleaved with project deletion."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

from django.db import close_old_connections
from django.db.models import signals
from organizations.tests.factories import OrganizationFactory
from projects.models import Project
from projects.tests.factories import ProjectFactory
from rest_framework.test import APIClient, APITestCase, APITransactionTestCase
from tasks.models import Task
from tasks.tests.factories import AnnotationDraftFactory, PredictionFactory, TaskFactory

LABEL_CONFIG = '<View><Text name="text" value="$text"/><Choices name="label" toName="text"><Choice value="yes"/></Choices></View>'
RESULT = [{'from_name': 'label', 'to_name': 'text', 'type': 'choices', 'value': {'choices': ['yes']}}]


class ProjectDeletionAnnotationTests(APITestCase):
    def setUp(self):
        self.organization = OrganizationFactory()
        self.user = self.organization.created_by
        self.client.force_authenticate(self.user)
        self.project = ProjectFactory(organization=self.organization, created_by=self.user, label_config=LABEL_CONFIG)
        self.task = TaskFactory(project=self.project, data={'text': 'signal regression'})

    def test_annotation_created_during_other_project_deletion_updates_counters(self):
        deleting_project = ProjectFactory(organization=self.organization, created_by=self.user)
        original_delete = Project.delete
        responses = []

        def delete_with_annotation(instance, *args, **kwargs):
            # Interleave a real annotation request at the deletion call boundary.
            responses.append(self.client.post(f'/api/tasks/{self.task.id}/annotations', {'result': RESULT}, format='json'))
            return original_delete(instance, *args, **kwargs)

        with patch.object(Project, 'delete', delete_with_annotation):
            deleted = self.client.delete(f'/api/projects/{deleting_project.id}')

        self.assertEqual(deleted.status_code, 204, deleted.content)
        self.assertEqual(responses[0].status_code, 201, responses[0].content)
        task = self.client.get(f'/api/tasks/{self.task.id}').json()
        self.assertEqual(task['total_annotations'], 1)
        self.assertTrue(task['is_labeled'])
        project = self.client.get(f'/api/projects/{self.project.id}').json()
        self.assertEqual(project['finished_task_number'], 1)

    def test_update_recovers_previously_incorrect_task_statistics(self):
        annotation = self.client.post(f'/api/tasks/{self.task.id}/annotations', {'result': RESULT}, format='json')
        self.assertEqual(annotation.status_code, 201, annotation.content)
        # Arrange the historical persisted state observed in production.
        Task.objects.filter(pk=self.task.id).update(total_annotations=0, is_labeled=False)
        before = self.client.get(f'/api/projects/{self.project.id}').json()
        self.assertEqual(before['finished_task_number'], 0)

        updated = self.client.patch(f'/api/annotations/{annotation.json()["id"]}', {'result': RESULT}, format='json')

        self.assertEqual(updated.status_code, 200, updated.content)
        task = self.client.get(f'/api/tasks/{self.task.id}').json()
        self.assertEqual(task['total_annotations'], 1)
        self.assertTrue(task['is_labeled'])
        self.assertEqual(self.client.get(f'/api/projects/{self.project.id}').json()['finished_task_number'], 1)

    def test_annotation_update_during_failed_other_project_deletion_keeps_signals_working(self):
        annotation = self.client.post(f'/api/tasks/{self.task.id}/annotations', {'result': RESULT}, format='json')
        self.assertEqual(annotation.status_code, 201)
        Task.objects.filter(pk=self.task.id).update(total_annotations=0, is_labeled=False)
        deleting_project = ProjectFactory(organization=self.organization, created_by=self.user)
        updates = []

        def failing_delete(instance, *args, **kwargs):
            updates.append(self.client.patch(f'/api/annotations/{annotation.json()["id"]}',
                                             {'result': RESULT}, format='json'))
            raise RuntimeError('simulated deletion failure')

        with patch.object(Project, 'delete', failing_delete):
            deleted = self.client.delete(f'/api/projects/{deleting_project.id}')

        self.assertEqual(deleted.status_code, 500)
        self.assertEqual(updates[0].status_code, 200)
        saved = self.client.get(f'/api/tasks/{self.task.id}').json()
        self.assertEqual(saved['total_annotations'], 1)
        self.assertTrue(saved['is_labeled'])
        self.assertEqual(self.client.get(f'/api/projects/{self.project.id}').json()['finished_task_number'], 1)
        following = TaskFactory(project=self.project, data={'text': 'after deletion failure'})
        created = self.client.post(f'/api/tasks/{following.id}/annotations', {'result': RESULT}, format='json')
        self.assertEqual(created.status_code, 201)
        self.assertEqual(self.client.get(f'/api/projects/{self.project.id}').json()['finished_task_number'], 2)

    def test_delete_cascades_tasks_annotations_predictions_and_drafts(self):
        annotation = self.client.post(f'/api/tasks/{self.task.id}/annotations', {'result': RESULT}, format='json')
        self.assertEqual(annotation.status_code, 201)
        prediction = PredictionFactory(task=self.task, project=self.project, result=RESULT)
        draft = AnnotationDraftFactory(task=self.task, user=self.user, result=RESULT)

        response = self.client.delete(f'/api/projects/{self.project.id}')

        self.assertEqual(response.status_code, 204, response.content)
        self.assertEqual(self.client.get(f'/api/projects/{self.project.id}').status_code, 404)
        self.assertEqual(self.client.get(f'/api/tasks/{self.task.id}').status_code, 404)
        self.assertEqual(self.client.get(f'/api/annotations/{annotation.json()["id"]}').status_code, 404)
        self.assertEqual(self.client.get(f'/api/predictions/{prediction.id}').status_code, 404)
        self.assertEqual(self.client.get(f'/api/drafts/{draft.id}').status_code, 404)


class ConcurrentProjectDeletionTests(APITransactionTestCase):
    def test_non_lifo_deletions_leave_subsequent_annotation_writes_working(self):
        organization = OrganizationFactory()
        user = organization.created_by
        projects = [ProjectFactory(organization=organization, created_by=user) for _ in range(2)]
        surviving = ProjectFactory(organization=organization, created_by=user, label_config=LABEL_CONFIG)
        task = TaskFactory(project=surviving, data={'text': 'surviving annotation'})
        entered = {project.pk: Event() for project in projects}
        release = {project.pk: Event() for project in projects}
        original_delete = Project.delete
        # Restore registrations after the test even when run against the broken baseline.
        model_signals = [signals.pre_init, signals.post_init, signals.pre_save, signals.post_save,
                         signals.pre_delete, signals.post_delete, signals.pre_migrate, signals.post_migrate]
        registrations = [(signal, list(signal.receivers)) for signal in model_signals]

        def paused_delete(instance, *args, **kwargs):
            entered[instance.pk].set()
            if not release[instance.pk].wait(15):
                raise TimeoutError('deletion interleaving was not released')
            return original_delete(instance, *args, **kwargs)

        def delete_project(project_id):
            close_old_connections()
            try:
                client = APIClient()
                client.force_authenticate(user)
                return client.delete(f'/api/projects/{project_id}')
            finally:
                close_old_connections()

        try:
            with patch.object(Project, 'delete', paused_delete), ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(delete_project, projects[0].pk)
                self.assertTrue(entered[projects[0].pk].wait(15))
                second = executor.submit(delete_project, projects[1].pk)
                self.assertTrue(entered[projects[1].pk].wait(15))
                release[projects[0].pk].set()
                self.assertEqual(first.result(timeout=15).status_code, 204)
                release[projects[1].pk].set()
                self.assertEqual(second.result(timeout=15).status_code, 204)

            self.client.force_authenticate(user)
            response = self.client.post(f'/api/tasks/{task.pk}/annotations', {'result': RESULT}, format='json')
            self.assertEqual(response.status_code, 201, response.content)
            saved = self.client.get(f'/api/tasks/{task.pk}').json()
            self.assertEqual(saved['total_annotations'], 1)
            self.assertTrue(saved['is_labeled'])
            self.assertEqual(self.client.get(f'/api/projects/{surviving.pk}').json()['finished_task_number'], 1)
        finally:
            for event in release.values():
                event.set()
            for signal, receivers in registrations:
                with signal.lock:
                    signal.receivers = receivers
                    signal.sender_receivers_cache.clear()

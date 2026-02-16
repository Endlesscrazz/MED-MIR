import { expect, test } from '@playwright/test';
import path from 'path';

test.describe('Med-MIR smoke tests', () => {
  test('home page loads and shows search UI', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /medical image retrieval/i })).toBeVisible();
    await expect(page.getByPlaceholder(/describe pathology/i)).toBeVisible();
  });

  test('fallback text query returns results', async ({ page }) => {
    await page.goto('/');

    const input = page.getByPlaceholder(/describe pathology/i);
    await input.fill('pneumonia');
    await input.press('Enter');

    await expect(page.getByText(/showing results for:/i)).toBeVisible();
    await expect(page.getByText(/cached/i)).toBeVisible();
  });

  test('image upload flow triggers image query state', async ({ page }) => {
    await page.goto('/');

    const fixturePath = path.resolve(__dirname, 'fixtures', 'sample.png');
    await page.locator('input[type="file"]').setInputFiles(fixturePath);

    await expect(page.getByText(/showing results for:/i)).toBeVisible();
    // Query label may be the immediate upload text or the post-inference text.
    await expect(
      page.getByText(/(image: sample\.png|visually similar to sample\.png)/i)
    ).toBeVisible();
  });

  test('metrics and hard-cases pages load', async ({ page }) => {
    await page.goto('/metrics');
    await expect(page.getByRole('heading', { name: /reliability metrics/i })).toBeVisible();

    await page.goto('/hard-cases');
    await expect(page.getByRole('heading', { name: /hard case analysis/i })).toBeVisible();
  });

  test('benchmark page runs and emits report', async ({ page }) => {
    await page.goto('/benchmark');
    await expect(page.getByRole('heading', { name: /browser benchmark/i })).toBeVisible();

    await page.getByTestId('run-benchmark').click();
    await expect(page.getByTestId('benchmark-report')).toBeVisible({ timeout: 180000 });
  });
});
